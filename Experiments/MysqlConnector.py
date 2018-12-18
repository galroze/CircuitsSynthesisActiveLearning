import mysql.connector
from sqlalchemy import exc
import sqlalchemy.pool as pool


def get_conn():
    return mysql.connector.connect(user='gal', host='127.0.0.1', database='test')


my_pool = pool.QueuePool(get_conn, max_overflow=5, pool_size=5)


def get_connection():
    return my_pool.connect()

