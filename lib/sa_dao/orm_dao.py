from sa_dao.sqlalchemy_dao import SqlAlchemyDAO
from sqlalchemy.sql import *
from sqlalchemy.sql import compiler
from sqlalchemy import cast, String, case
import sqlalchemy.orm as orm
from sqlalchemy.orm import aliased, class_mapper
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.orm.properties import RelationshipProperty
from sqlalchemy.orm import sessionmaker
import re
import types
import logging


class ORM_DAO(SqlAlchemyDAO):

    def __init__(self, session=None, **kwargs):
        SqlAlchemyDAO.__init__(self, connection=session.connection(), **kwargs)
        self.session = session

    def get_session_w_external_transaction(self):
        con = self.session.bind.connect()
        trans = con.begin()
        session = sessionmaker(bind=connection)()
        return {
            'connection': con,
            'transaction': trans,
            'session': session
        }

    def join_(self, *args, **kwargs):
        return orm.join(*args, **kwargs)

    def assemble_query(self, selections=[], froms=[], wheres=[], group_bys=[],
                       order_bys=[]):
        q = self.session.query(*selections)\
                .select_from(*froms)\
                .filter(*wheres)\
                .group_by(*group_bys)\
                .order_by(*order_bys)
        return q

    def get_registered_source(self, source_registry=None, source_def=None,
                              **kwargs):
        source_def = self.prepare_source_def(source_def)

        node = source_registry['nodes'].get(source_def['ID'])
        if not node:
            # If 'source' is a dict , we assume it's a query object and process it.
            if isinstance(source_def['SOURCE'], dict):
                source = self.get_query(source_def['SOURCE']).alias(source_def['ID'])
                node = {
                    'source': source,
                    'children': {}
                }
            # Otherwise we process the source path...
            else:
                parts = source_def['SOURCE'].split('__')

                # Register dependencies and add to join tree.
                parent_node = source_registry['join_tree']
                if len(parts) < 2:
                    source_id = '__'.join(parts)
                    source = aliased(self.schema['sources'][source_id])
                    node = {
                        'source': source,
                        'children': {}
                    }
                    source_registry['nodes'][source_id] = node
                    parent_node['children'][source_id] = node
                else:
                    for i in range(1, len(parts) + 1):
                        parent_id = '__'.join(parts[:i])
                        if source_registry['nodes'].has_key(parent_id):
                            node = source_registry['nodes'][parent_id]
                        else:
                            if i == 1:
                                source = aliased(self.schema['sources'][parent_id])
                                parent_attr = parent_id
                            else:
                                grandparent_id = '__'.join(parts[:i-1])
                                parent_attr = parts[i-1]
                                mapped_grandparent = source_registry['nodes'].get(
                                    grandparent_id)['source']
                                parent_prop = class_mapper(
                                    mapped_grandparent._AliasedClass__target
                                ).get_property(parent_attr)
                                if isinstance(parent_prop, RelationshipProperty):
                                    source = aliased(parent_prop.mapper.class_)
                            node = {
                                'source': source,
                                'children': {}
                            }
                            source_registry['nodes'][parent_id] = node
                            parent_node['children'][parent_attr] = node
                        parent_node = node

        return node['source']

    def get_registered_entity(self, source_registry=None, entity_registry=None,
                              entity_def=None, token_registry=None, **kwargs):

        entity_def = self.prepare_entity_def(entity_def)

        # Map and register entity if not in the registry.
        if not entity_registry.has_key(entity_def['ID']):

            mapped_entities = {}

            # First validate the expression.  This will throw an error
            # if the expression is invalid.
            self.expression_validator.validate_expression(entity_def['EXPRESSION'])

            # Replace entity tokens in expression w/ mapped entities.
            # This will be called for each token match.
            def replace_token_with_mapped_entity(m):
                token = m.group(1)
                parts = token.split('__')
                parts = parts[1:] # first is blank, due to initial '__'
                attr_id = parts[-1]
                source_def = '__'.join(parts[:-1])
                if source_def:
                    # Special tokens source.
                    # This is intended to allow for substitution of key
                    # entities inside queries.
                    if source_def == '_TOKENS':
                        token_def = token_registry.get(attr_id)
                        token_entity = self.get_registered_entity(
                            source_registry=source_registry,
                            entity_registry=entity_registry,
                            token_registry=token_registry,
                            entity_def=token_def).element
                        mapped_entities[token] = token_entity
                    else:
                        source = self.get_registered_source(
                            source_registry, source_def)
                        mapped_entities[token] = getattr(source, attr_id)
                else:
                    mapped_entities[token] = self.get_registered_source(
                        source_registry, attr_id)
                mapped_entities[token] = self.alter_col(mapped_entities[token])
                return "mapped_entities['%s']" % token

            expression_code = re.sub(r'\b(__(\w+))+\b', 
                                     replace_token_with_mapped_entity, 
                                     entity_def['EXPRESSION'])

            # Evaluate and label.
            mapped_entity = self.eval_expression_code(
                expression_code, globals(), locals()
            )
            if isinstance(mapped_entity, AliasedClass): 
                mapped_entity._sa_label_name = entity_def['ID']
            else:
                mapped_entity = mapped_entity.label(entity_def['ID'])

            # Register.
            entity_registry[entity_def['ID']] = mapped_entity

        return entity_registry[entity_def['ID']]

    def get_result_cursor(self, q):
        self.cursorify_query(q)
        return q

    def cursorify_query(self, q):
        def fetchall(self):
                return self.all()
        q.fetchall = types.MethodType(fetchall, q)

        def fetchone(self):
            return self.one()
        q.fetchone = types.MethodType(fetchone, q)

    def query_to_raw_sql(self, q, dialect=None):
        return super(ORM_DAO, self).query_to_raw_sql(
            q.statement, dialect=dialect)

    def save(self, obj, commit=True):
        self.session.add(obj)
        if commit:
            self.commit()

    def save_all(self, objs, commit=True):
        self.session.add_all(objs)
        if commit:
            self.commit()

    def commit(self):
        self.session.commit()

    def get_table_for_class(self, clazz):
        return class_mapper(clazz).mapped_table

    def get_batched_results(self, q, batch_size):
        """ Return an iterator that batches query results.
        This is usually used in order to avoid overloading memory.
        Ideally we would use window functions, but not all dbs support this.
        """
        total = q.count()
        returned = 0
        batch = None
        # While < total_rows, if no batch, get a new batch.
        # for item in batch, yield.
        while returned < total:
            if not batch:
                batch = q.limit(batch_size).offset(returned)
            for row in batch:
                returned += 1
                if (returned % batch_size) == 0:
                    batch = None
                yield row

    def query(self, query_def, format_='result_cursor', **kwargs):
        q = self.get_query(query_def, **kwargs)
        if format_ == 'result_cursor':
            return self.get_result_cursor(q)
        elif format_ == 'query_obj':
            return q


    def save_dicts(self, source_id, dicts, batch_size=10000,
                   commit=True, **kwargs):

        table = self.get_table_for_class(self.schema['sources'][source_id])

        if batch_size is None:
            self.session.execute(table.insert(), dicts)
        else:
            batch = []
            batch_counter = 1
            for d in dicts:
                # If batch is at batch size, process the batch w/out committing.
                if (batch_counter % batch_size) == 0: 
                    self.save_dicts(
                        source_id,
                        batch, 
                        batch_size=None,
                        commit=False
                    )
                    batch = []
                batch.append(d)
                batch_counter += 1

            # Save any remaining batch items.
            if batch: 
                self.save_dicts(
                    source_id, 
                    batch, 
                    batch_size=None, 
                    commit=False,
                )


        # Commit if commit is true.
        if commit: 
            self.session.commit()
