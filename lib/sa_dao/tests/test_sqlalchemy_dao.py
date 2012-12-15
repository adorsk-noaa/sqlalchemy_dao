import unittest
from sa_dao.tests.basetest import BaseTest
from sa_dao.sqlalchemy_dao import SqlAlchemyDAO
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (Table, Column, ForeignKey, ForeignKeyConstraint, 
                        Integer, String, Float, Index)
from sqlalchemy.orm import (relationship)


class SqlAlchemyDAOTestCase(BaseTest):

    def setUp(self):
        super(SqlAlchemyDAOTestCase, self).setUp()
        self.schemas = {
            'schema1':  self.setUpSchemaAndData1()
        }

    def xtest_token_query(self):
        schema = self.schemas['schema1']
        dao = SqlAlchemyDAO(connection=self.connection, schema=schema)
        # Generate data.
        self.connection.execute(
            schema['sources']['substrate'].insert(), 
            [{'id': 'S1', 'label': 's1'}]
        )
        tkn1 = {'ID': 'tkn1', 'EXPRESSION': '__substrate__id'}
        token_registry = {'tkn1': tkn1}
        query_def = {
            'ID': 'test',
            'SELECT': [
                {'ID': 'tkn_id', 'EXPRESSION': '___TOKENS__tkn1'}
            ]
        }
        dao.execute_queries([query_def], token_registry=token_registry)

    def test_keyed_query(self):
        schema = self.schemas['schema1']
        dao = SqlAlchemyDAO(connection=self.connection, schema=schema)
        key_def = {
            'LABEL_ENTITY': {
                'ID': 'substrate_name'
            }, 
            'QUERY': {
                'SELECT': [
                    {'EXPRESSION': '__substrate__id', 'ID': 'substrate_id'}, 
                    {'EXPRESSION': '__substrate__label', 'ID': 'substrate_name'}
                ]
            }, 
            'KEY_ENTITY': {
                'ID': 'substrate_id', 'EXPRESSION': '__result__id'
            }
        }

        query_def = {
            'GROUP_BY': [], 
            'FROM': [
                {
                    'SOURCE': {
                        'GROUP_BY': [
                            {'ID': 'cell_id'}, 
                            {'ID': 'substrate_id'}
                        ], 
                        'WHERE': [
                            [{'TYPE': 'ENTITY', 'EXPRESSION': '__result__t', 'ID': 't'}, '==', 2]
                        ], 
                        'ID': 'inner', 
                        'SELECT': [
                            {'EXPRESSION': '__result__cell_id', 'ID': 'cell_id'}, 
                            #{'EXPRESSION': '__result__substrate_id', 'ID': 'substrate_id'}
                            {'EXPRESSION': '___TOKENS__KEY', 'ID': 'substrate_id'}
                        ], 
                        'SELECT_GROUP_BY': True
                    }, 
                    'ID': 'inner'
                }, 
                {
                    'SOURCE': 'cell', 
                    'JOINS': [
                        ['inner', [{'TYPE': 'ENTITY', 'EXPRESSION': '__inner__cell_id'}, '==', {'TYPE': 'ENTITY', 'EXPRESSION': '__cell__id'}]]
                    ]
                }, 
                {
                    'SOURCE': 'substrate', 
                    'JOINS': [
                        ['inner', [{'TYPE': 'ENTITY', 'EXPRESSION': '__inner__substrate_id'}, '==', {'TYPE': 'ENTITY', 'EXPRESSION': '__substrate__id'}]]
                    ]
                }
            ], 
            'ID': 'outer', 
            'SELECT': [
                {'EXPRESSION': 'func.sum(__cell__area)', 'ID': 'sum_cell_area'}, 
                {'EXPRESSION': '__substrate__label', 'ID': 'substrate_name'}
            ]
        }

        dao.get_keyed_results(key_def=key_def, query_defs=[query_def])

    def setUpSchemaAndData1(self):
        schema = {}
        sources = {}

        metadata = MetaData()

        # Times go in their own table to speed up time queries.
        # Otherwise we have to scan all results to just get a list of times.
        sources['time'] = Table('time', metadata,
                Column('id', Integer, primary_key=True),
                )

        sources['cell'] = Table('cell', metadata,
                Column('id', Integer, primary_key=True),
                Column('z', Float),
                Column('area', Float),
                )

        sources['energy'] = Table('energy', metadata,
                Column('id', String, primary_key=True),
                Column('label', String),
                )

        sources['substrate'] = Table('substrate', metadata,
                Column('id', String, primary_key=True),
                Column('label', String),
                )

        sources['feature'] = Table('feature', metadata,
                Column('id', String, primary_key=True),
                Column('label', String),
                Column('category', String),
                )

        sources['gear'] = Table('gear', metadata,
                Column('id', String, primary_key=True),
                Column('label', String),
                )

        sources['result']= Table('result', metadata,
                Column('id', Integer, primary_key=True),
                Column('t', Integer),
                Column('energy_id', String, ForeignKey('energy.id')),
                Column('cell_id', Integer, ForeignKey('cell.id')),
                Column('gear_id', String, ForeignKey('gear.id')),
                Column('substrate_id', String, ForeignKey('substrate.id')),
                Column('feature_id', String, ForeignKey('feature.id')),
                Column('a', Float),
                Column('x', Float),
                Column('y', Float),
                Column('z', Float),
                Column('znet', Float),
                )
        # Create time/key indices. These are essential for filtering in a reasonable
        # amount of time.
        for col in ['energy_id', 'cell_id', 'gear_id', 'substrate_id', 'feature_id']:
            Index('idx_t_%s' % col, sources['result'].c.t, sources['result'].c[col])

        metadata.create_all(self.connection)

        # This dictionary contains the schema objects GeoRefine will use.
        schema = {
            'sources': sources,
            'metadata': metadata,
        }

        """
        test1_rows = []
        for i in range(5):
            test1_row = {}
            test1_rows.append(test1_row)
        trans = self.connection.begin()
        self.connection.execute(sources['test1'].insert(), test1_rows)
        trans.commit()
        """

        return schema

if __name__ == '__main__':
    unittest.main()
