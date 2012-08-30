import unittest
from sa_dao.tests.basetest import BaseTest
from sa_dao.orm_dao import ORM_DAO
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (Table, Column, ForeignKey, ForeignKeyConstraint, 
                        Integer, String, Float)
from sqlalchemy.orm import (relationship)


class ORM_DAO_Test(BaseTest):

    def setUp(self):
        super(ORM_DAO_Test, self).setUp()
        self.schemas = {
            'schema1':  self.setUpSchemaAndData1()
        }

    def xtest_join_query(self):
        schema = self.schemas['schema1']
        dao = ORM_DAO(session=self.session, schema=schema)
        simple_q = {
            'ID': 'simple_q',
            'SELECT': ['{{TestClass1.id}}'],
            'WHERE': [['{{TestClass1.children.id}}', '==', 1]],
        }
        results = dao.execute_queries(query_defs=[simple_q])

    def xtest_obj_query(self):
        schema = self.schemas['schema1']
        dao = ORM_DAO(session=self.session, schema=schema)
        q = {
            'ID': 'obj_q',
            'SELECT': ['{{TestClass1}}'],
        }
        results = dao.execute_queries(query_defs=[q])

    def test_combined_query(self):
        schema = self.schemas['schema1']
        dao = ORM_DAO(session=self.session, schema=schema)
        q = {
            'ID': 'obj_q',
            'SELECT': [
                {'ID': 'foo', 'EXPRESSION': '{{TestClass1}}'}, 
                '{{TestClass1.id}}'],
        }
        results = dao.execute_queries(query_defs=[q])

    def setUpSchemaAndData1(self):
        schema = {}
        sources = {}

        Base = declarative_base()

        # TestClass1
        class TestClass1(Base):
            __tablename__ = 'testclass1'
            id = Column(Integer, primary_key=True)
            name = Column(String)
            children = relationship('TestClass2')

            def __init__(self, id=None, name=None, children=[]):
                self.id = id
                self.name = name
                self.children = children

        sources['TestClass1'] = TestClass1

        # TestClass2
        class TestClass2(Base):
            __tablename__ = 'testclass2'
            id = Column(Integer, primary_key=True)
            name = Column(String)
            parent_id = Column(Integer, ForeignKey(TestClass1.__table__.c.id))

            def __init__(self, id=None, name=None):
                self.id = id
                self.name = name

        sources['TestClass2'] = TestClass2

        # Save classes to schema.
        schema['sources'] = sources

        # Setup tables.
        Base.metadata.create_all(self.session.bind)

        # Generate data.
        tc1s = []
        for i in range(5):
            tc1 = TestClass1(
                name="tc1_%s" % i,
                children=[ TestClass2(name="tc2_%s" % j) for j in 
                         [1,2]]
            )
            tc1s.append(tc1)
        self.session.add_all(tc1s)
        self.session.commit()

        return schema

if __name__ == '__main__':
    unittest.main()
