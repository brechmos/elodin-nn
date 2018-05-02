from flask import (Flask, request, Blueprint, make_response,
        current_app, jsonify, abort)
import imageio
from io import BytesIO
import requests
import logging
import os.path
from tlapi.utils import gzipped
from tlapi.data import processing as data_processing

FORMAT = '%(levelname)-8s %(asctime)-15s %(name)-10s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('data')
log.setLevel(logging.DEBUG)

blueprint = Blueprint('asdf', __name__, url_prefix='/data')


@blueprint.route('/', methods=['GET'])
@blueprint.route('/<pk>', methods=['GET'])
@gzipped
def get(pk=None):
    log.info('get with pk {}'.format(pk))
    data = data_processing.get(current_app.db, pk)
    return jsonify(data)


@blueprint.route('/array/<pk>', methods=['GET'])
@gzipped
def get_array(pk=None):
    log.info('get_array with pk {}'.format(pk))

    nparray = data_processing.get_array(current_app.db, pk)

    return jsonify(nparray.tolist())


@blueprint.route('/save', methods=['POST'])
@gzipped
def save():
    log.info('save')

    payload = request.get_json()

    toreturn = data_processing.save(current_app.db, payload)

    return jsonify(toreturn)
