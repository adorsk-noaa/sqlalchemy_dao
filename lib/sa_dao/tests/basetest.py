import unittest
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table, Column, ForeignKey, ForeignKeyConstraint, Integer, String, Float

class BaseTest(unittest.TestCase):

    def setUp(self):
        self.engine = create_engine(
            'sqlite://',
            convert_unicode=True,
        )
        self.Session = sessionmaker()
        self.connection = self.engine.connect()

        # begin a non-ORM transaction
        self.trans = self.connection.begin()

        # bind an individual Session to the connection
        self.session = self.Session(bind=self.connection)

    def tearDown(self):
        # rollback - everything that happened with the
        # Session above (including calls to commit())
        # is rolled back.
        self.trans.rollback()
        self.session.close()

if __name__ == '__main__':
    unittest.main()
