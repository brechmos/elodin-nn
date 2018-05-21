import os

from transfer_learning.data import Data
from transfer_learning.cutout import Cutout
from transfer_learning.fingerprint import Fingerprint
from transfer_learning.similarity import Similarity

# Mongo
import pymongo
from bson import ObjectId

# Blitzdb
from blitzdb import FileBackend, Document
import shutil

# Unqlite
import unqlite

from .tl_logging import get_logger

import logging
log = get_logger('database')


def get_database(database_type, *args, **kwargs):
    dbcls = None
    for db in Database.__subclasses__():
        if db._database_type == database_type:
            dbcls = db
    return dbcls(*args, **kwargs)


def get_factory(table):
    if table == 'data':
        return Data.factory
    elif table == 'cutout':
        return Cutout.factory
    elif table == 'fingerprint':
        return Fingerprint.factory
    elif table == 'similarity':
        return Similarity.factory


class Database:
    """
    This is the base class to force class definitions.
    """

    def __init__(self):
        self._single_thread_required = True

    @property
    def single_thread_required(self):
        return self._single_thread_required

    def open(self):
        raise NotImplementedError("Please Implement this method")

    def save(self, table, data):
        raise NotImplementedError("Please Implement this method")

    def find(self, table, key=None, db=None):
        raise NotImplementedError("Please Implement this method")

    def count(self, table):
        raise NotImplementedError("Please Implement this method")

    def update(self, table, key, data):
        raise NotImplementedError("Please Implement this method")

    def close(self):
        raise NotImplementedError("Please Implement this method")

    def delete_database(self):
        raise NotImplementedError("Please Implement this method")


class Mongo(Database):

    key = '_id'
    _database_type = 'mongo'

    def _convert_objectid(self, dct):
        return {k: v if not isinstance(v, ObjectId) else str(v) for k, v in dct.items()}

    def __init__(self, hostname, port=27017):
        self._hostname = hostname
        self._port = port
        self._single_thread_required = False

        self._mongo = pymongo.MongoClient(self._hostname, self._port)
        self._database = self._mongo.database

    def save(self, table, data):

        # Convert to dict if not one already
        if not isinstance(data, dict):
            data = data.save()
        
        mongo_table = getattr(self._database, table)
        data.update({'_id': data['uuid']})
        tt = mongo_table.insert_one(data)
        return str(tt.inserted_id)

    def find(self, table, key=None):
        mongo_table = getattr(self._database, table)

        factory = get_factory(table)

        if key is None:
            toreturn = mongo_table.find({})
            return [factory(self._convert_objectid(x)) for x in toreturn]
        elif isinstance(key, list):
            data = mongo_table.find({'_id': {'$in': [ObjectId(x) for x in key]}})
            return [factory(self._convert_objectid(x)) for x in data]
        else:
            toreturn = mongo_table.find_one({'_id': ObjectId(key)})
            if toreturn is None:
                toreturn = {}
            return factory(self._convert_objectid(toreturn))

    def count(self, table):
        mongo_table = getattr(self._database, table)
        return mongo_table.count()

    def update(self, table, key, data):
        mongo_table = getattr(self._database, table)
        mongo_table.replace_one({'_id': ObjectId(key)}, data)
        return

    def close(self):
        pass

    def delete_database(self):
        self._mongo.drop_database('database')


class BlitzDB(Database):

    key = 'pk'
    _database_type = 'blitzdb'

    class Data(Document):
        pass

    class Cutout(Document):
        pass

    class Fingerprint(Document):
        pass

    class Similarity(Document):
        pass

    def _get_table(self, table_name):
        if table_name == 'data':
            return BlitzDB.Data
        elif table_name == 'cutout':
            return BlitzDB.Cutout
        elif table_name == 'fingerprint':
            return BlitzDB.Fingerprint
        elif table_name == 'similarity':
            return BlitzDB.Similarity
        else:
            log.error('BAD TABLE NAME {}'.format(table_name))

    def __init__(self, filename):
        self._filename = filename
        self._backend = FileBackend(self._filename)

    def save(self, table, data):

        # Convert to dict if not one already
        if not isinstance(data, dict):
            data = data.save()

        blitz_table = self._get_table(table)
        data.update({'pk': data['uuid']})
        save_id = self._backend.save(blitz_table(data))
        self._backend.commit()
        return save_id['pk']

    def find(self, table, key=None):
        blitz_table = self._get_table(table)
        factory = get_factory(table)

        if key is None:
            return [factory(dict(x), db=self) for x in self._backend.filter(blitz_table, {})]
        elif isinstance(key, list):
            return [factory(dict(x), db=self) for x in self._backend.filter(blitz_table, {'pk': {'$in': key}})]
        else:
            return factory(dict(self._backend.get(blitz_table, {'pk': key})), db=self)

    def count(self, table):
        blitz_table = self._get_table(table)
        return len(self._backend.filter(blitz_table, {}))

    def update(self, table, key, data):
        blitz_table = self._get_table(table)
        entry = self._backend.get(blitz_table, {'pk': key})
        for k, v in data.items():
            setattr(entry, k, v)
        entry.save()
        self._backend.commit()

    def close(self):
        pass

    def delete_database(self):
        shutil.rmtree(self._filename)


class UnQLite(Database):
    """
    The UnQLite database will be behind a flask layer
    as UnQLite is not multi-threaded, ACID etc.
    """

    key = '__id'
    _database_type = 'unqlite'

    def __init__(self, filename):
        """
        Initialize the database and flask app
        """
        log.info('')

        self._single_thread_required = False

        # Initialize the input parameters
        self._filename = filename

        self._db = unqlite.UnQLite(self._filename)
        log.debug('Opened unqlite database file {} {}'.format(self._filename, self._db))

        # Setup the unqlite database collections
        log.debug('Setting up data collection')
        self._data = self._db.collection('data')
        if not self._data.exists():
            log.debug('  ... doesn''t exist so creating.')
            self._data.create()

        self._cutout = self._db.collection('cutout')
        log.debug('Setting up cutout collection')
        if not self._cutout.exists():
            log.debug('  ... doesn''t exist so creating.')
            self._cutout.create()

        self._fingerprint = self._db.collection('fingerprint')
        log.debug('Setting up fingerprint collection')
        if not self._fingerprint.exists():
            log.debug('  ... doesn''t exist so creating.')
            self._fingerprint.create()

        self._similarity = self._db.collection('similarity')
        log.debug('Setting up similarity collection')
        if not self._similarity.exists():
            log.debug('  ... doesn''t exist so creating.')
            self._similarity.create()

    def _get_table(self, table_name):
        """
        Convert between a string table name and a unqlite collection.
        """
        if table_name == 'data':
            return self._data
        elif table_name == 'cutout':
            return self._cutout
        elif table_name == 'fingerprint':
            return self._fingerprint
        elif table_name == 'similarity':
            return self._similarity
        else:
            log.error('BAD TABLE NAME {}'.format(table_name))

    def delete_database(self):
        if os.path.isdir(self._filename):
            shutil.rmtree(self._filename)

    def save(self, table, data):
        """
        Save the data in the unqlite database.
        data is also sent through POST

        :param table: string representation of the table/collection name
        """
        log.info('saving to {}'.format(table))

        # Convert to dict if not one already
        if not isinstance(data, dict):
            data = data.save()
        
        collection = self._get_table(table)
        with self._db.transaction():
            try:
                collection.store(data)
            except Exception as e:
                log.error(e)

        return collection.last_record_id()

    def find(self, table, key=None):
        """
        Find data in the unqlite database.

        :param table: string representation of the table/collection name
        """
        log.info('Searching for data in {} with key {}'.format(table, key))
        collection = self._get_table(table)
        log.debug('Collection is {}'.format(collection))
        if key is None:
            return self._convert([x for x in collection.all()])
        elif isinstance(key, list):
            return self._convert([x for x in collection.filter(lambda obj: str(obj['__id']) in str(key))])
        else:
            data = self._convert(collection.filter(lambda obj: str(obj['__id']) == str(key)))[0]
            return data

    def count(self, table):
        """
        Count the number of elements in the table/collection.

        :param table: string representation of the table/collection name
        """
        log.info('flask unqlite count table {}'.format(table))
        collection = self._get_table(table)
        return len(collection)

    def update(self, table, key, data):
        """
        Update data in the unqlite table/collection.
        Data and key are sent through the POST json data.

        :param table: string representation of the table/collection name
        """
        log.info('flask unqlite update table {}'.format(table))
        collection = self._get_table(table)
        data['__id'] = key
        with self._db.transaction():
            collection.update(key, data)

        return key

    def _convert(self, input):
        """
        The unqlite data (values) are stored as bytes, not strings
        so we need to decode the data before using jsonify.
        """
        if isinstance(input, dict):
            return {self._convert(key): self._convert(value) for key, value in input.items()}
        elif isinstance(input, list):
            return [self._convert(element) for element in input]
        elif isinstance(input, bytes):
            return input.decode('utf-8')
        else:
            return input

    def close(self):
        self._db.close()
