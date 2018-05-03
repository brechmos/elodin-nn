import logging
import os

# Mongo
import pymongo
from bson import ObjectId

# Blitzdb
from blitzdb import FileBackend, Document
import shutil

# Unqlite
import requests
import threading
import unqlite
import time
import numpy as np

FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(funcName)10s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('database')
log.setLevel(logging.INFO)


def get_database(database_type, *args, **kwargs):
    dbcls = None
    for db in Database.__subclasses__():
        if db._database_type == database_type:
            dbcls = db
    return dbcls(*args, **kwargs)


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

    def find(self, table, key=None):
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
        mongo_table = getattr(self._database, table)
        data.update({'_id': data['uuid']})
        tt = mongo_table.insert_one(data)
        return str(tt.inserted_id)

    def find(self, table, key=None):
        mongo_table = getattr(self._database, table)
        if key is None:
            toreturn = mongo_table.find({})
            return [self._convert_objectid(x) for x in toreturn]
        elif isinstance(key, list):
            data = mongo_table.find({'_id': {'$in': [ObjectId(x) for x in key]}})
            return [self._convert_objectid(x) for x in data]
        else:
            toreturn = mongo_table.find_one({'_id': ObjectId(key)})
            if toreturn is None:
                toreturn = {}
            return self._convert_objectid(toreturn)

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

    class Fingerprint(Document):
        pass

    class Similarity(Document):
        pass

    def _get_table(self, table_name):
        if table_name == 'data':
            return BlitzDB.Data
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
        blitz_table = self._get_table(table)
        data.update({'pk': data['uuid']})
        save_id = self._backend.save(blitz_table(data))
        self._backend.commit()
        return save_id['pk']

    def find(self, table, key=None):
        blitz_table = self._get_table(table)
        if key is None:
            return [dict(x) for x in self._backend.filter(blitz_table, {})]
        elif isinstance(key, list):
            return [dict(x) for x in self._backend.filter(blitz_table, {'pk': {'$in': key}})]
        else:
            return dict(self._backend.get(blitz_table, {'pk': key}))

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


#class UnQLite(Database):
#    """
#    The UnQLite database will be behind a flask layer
#    as UnQLite is not multi-threaded, ACID etc.
#    """
#
#    key = '__id'
#    _database_type = 'unqilte'
#
#    # Singleton instance
#    __instance = None
#
#    def __new__(cls, val):
#        log.info('val'.format(val))
#        if UnQLite.__instance is None:
#            UnQLite.__instance = object.__new__(cls)
#        UnQLite.__instance.val = val
#        return UnQLite.__instance
#
#    def __init__(self, filename, host='127.0.0.1', port=5555):
#        """
#        Initialize the database and flask app
#        """
#        log.info('')
#
#        self._single_thread_required = False
#
#        self._host = host if host.startswith('http://') else 'http://' + host
#        self._port = port
#
#        # Initialize the input parameters
#        self._filename = filename
#
#        # Jitter in case several are starting up at the same time.
#        # this might not be needed and might actually be dumb.
#        time.sleep(2 * np.random.rand())
#        if not self.is_up():
#            log.info('  database server is not up so starting it now')
#            self._t = threading.Thread(target=self._start_flask, args=())
#            self._t.daemon = True
#            self._t.start()
#
#            # Short wait...
#            time.sleep(2)
#
#        # Check to see if it is up now....
#        if self.is_up():
#            log.info('Now the database is up...')
#        else:
#            log.error('The database is still down...')
#            raise Exception('Database is not up')
#
#    def _get_table(self, table_name):
#        """
#        Convert between a string table name and a unqlite collection.
#        """
#        if table_name == 'data':
#            return self._data
#        elif table_name == 'fingerprint':
#            return self._fingerprint
#        elif table_name == 'similarity':
#            return self._similarity
#        else:
#            log.error('BAD TABLE NAME {}'.format(table_name))
#
#    def save(self, table, data):
#        """
#        Post the data to save in the database.
#
#        :param self: - self
#        :param table: - string representation of the table
#        :param data: - python dict to store
#        """
#        url = '{}:{}/save/{}/'.format(self._host, self._port, table)
#        resp = requests.post(url, json=data)
#        return resp.json()
#
#    def find(self, table, key=None):
#        """
#        Find the data to save in the database.
#
#        :param self: - self
#        :param table: - string representation of the table
#        :param key: - string representation of the key
#        """
#        url = '{}:{}/find/{}/'.format(self._host, self._port, table)
#
#        log.info('unqlite find from {} with key {}'.format(url, key))
#        resp = requests.post(url, json={'key': key})
#        return resp.json()
#
#    def count(self, table):
#        """
#        Count the number of elements in the collection.
#
#        :param self: - self
#        :param table: - string representation of the table
#        """
#        url = '{}:{}/count/{}/'.format(self._host, self._port, table)
#        log.info('unqlite count from {}'.format(url))
#        resp = requests.get(url)
#        return resp.json()
#
#    def update(self, table, key, data):
#        """
#        Count the number of elements in the collection.
#
#        :param self: - self
#        :param table: - string representation of the table
#        :param key: - string representation of the key
#        :param data: - python dict of the data
#        """
#        url = '{}:{}/update/{}/'.format(self._host, self._port, table)
#        log.info('unqlite save to {}'.format(url))
#        payload = {
#            'data': data,
#            'key': key
#        }
#        resp = requests.post(url, json=payload)
#        return resp.json()
#
#    def is_up(self):
#        """
#        Check to see if the database site is up or not.
#        """
#        log.info('')
#        try:
#            code = requests.get('{}:{}/up'.format(self._host, self._port), timeout=1.0).status_code
#            if code == 200:
#                log.debug('  db server is up (code {})'.format(code))
#                return True
#            elif code == 404:
#                log.debug('  db server is down (code {})'.format(code))
#                return False
#            else:
#                raise Exception('Problem checking existence of database server {}'.format(code))
#        except Exception as e:
#            log.debug('  db server is down (exception {})'.format(str(e)))
#            return False
#
#    def close(self):
#        pass
#
#    def delete_database(self):
#        if os.path.isdir(self._filename):
#            shutil.rmtree(self._filename)
#
#    # --------------
#    #  Flask section
#
#    def _start_flask(self):
#        """
#        Start the flask app - this should be run in a thread
#        as it should just live in the background.
#        """
#
#        self._db = unqlite.UnQLite(self._filename)
#
#        # Setup the unqlite database collections
#        self._data = self._db.collection('data')
#        if not self._data.exists():
#            self._data.create()
#
#        self._fingerprint = self._db.collection('fingerprint')
#        if not self._fingerprint.exists():
#            self._fingerprint.create()
#
#        self._similarity = self._db.collection('similarity')
#        if not self._similarity.exists():
#            self._similarity.create()
#
#        # Setup the flask app
#        self._flask_app = Flask(__name__)
#        self._flask_app.secret_key = 'laskdjflasjkdflkj23lrjljks'
#        self._flask_app.add_url_rule('/up/', 'up', self.up)
#        self._flask_app.add_url_rule('/save/<table>/', 'save', self._flask_save, methods=['POST'])
#        self._flask_app.add_url_rule('/find/<table>/', 'find', self._flask_find, methods=['GET', 'POST'])
#        self._flask_app.add_url_rule('/count/<table>/', 'count', self._flask_count, methods=['GET'])
#        self._flask_app.add_url_rule('/update/<table>/', 'update', self._flask_update, methods=['POST'])
#
#        host = self._host
#        if 'http://' in host:
#            host = host.replace('http://', '')
#        if 'https://' in host:
#            host = host.replace('https://', '')
#        log.info('starting flask with host {} and port {}'.format(host, self._port))
#
#        self._flask_app.run(host=host, port=self._port)
#
#    def up(self):
#        return jsonify('Up')
#
#    def _flask_save(self, table):
#        """
#        Save the data in the unqlite database.
#        data is also sent through POST
#
#        :param table: string representation of the table/collection name
#        """
#        data = request.json
#
#        collection = self._get_table(table)
#        with self._db.transaction():
#            collection.store(data)
#        return jsonify(collection.last_record_id())
#
#    def _flask_find(self, table):
#        """
#        Find data in the unqlite database.
#
#        :param table: string representation of the table/collection name
#        """
#        try:
#            key = request.json['key']
#        except Exception as e:
#            log.warning('Exception {}'.format(e))
#            key = None
#
#        collection = self._get_table(table)
#        if key is None:
#            return jsonify(self._convert([x for x in collection.all()]))
#        elif isinstance(key, list):
#            return jsonify(self._convert([x for x in collection.filter(lambda obj: str(obj['__id']) in str(key))]))
#        else:
#            data = self._convert(collection.filter(lambda obj: str(obj['__id']) == str(key)))[0]
#            return jsonify(data)
#
#    def _flask_count(self, table):
#        """
#        Count the number of elements in the table/collection.
#
#        :param table: string representation of the table/collection name
#        """
#        log.info('flask unqlite count table {}'.format(table))
#        collection = self._get_table(table)
#        return jsonify(len(collection))
#
#    def _flask_update(self, table):
#        """
#        Update data in the unqlite table/collection.
#        Data and key are sent through the POST json data.
#
#        :param table: string representation of the table/collection name
#        """
#        log.info('flask unqlite update table {}'.format(table))
#        payload = request.json
#        data = payload['data']
#        key = payload['key']
#
#        collection = self._get_table(table)
#        data['__id'] = key
#        with self._db.transaction():
#            collection.update(key, data)
#
#        return jsonify(key)
#
#    def _convert(self, input):
#        """
#        The unqlite data (values) are stored as bytes, not strings
#        so we need to decode the data before using jsonify.
#        """
#        if isinstance(input, dict):
#            return {self._convert(key): self._convert(value) for key, value in input.items()}
#        elif isinstance(input, list):
#            return [self._convert(element) for element in input]
#        elif isinstance(input, bytes):
#            return input.decode('utf-8')
#        else:
#            return input
