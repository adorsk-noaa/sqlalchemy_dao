from sa_dao.sqlalchemy_dao import SqlAlchemyDAO
from sqlalchemy.sql import *
from sqlalchemy.sql import compiler
from sqlalchemy import cast, String, case
import sqlalchemy.orm as orm
from sqlalchemy.orm import aliased, class_mapper
from sqlalchemy.orm.util import AliasedClass
from sqlalchemy.orm.properties import RelationshipProperty
import re
import types


class ORM_DAO(SqlAlchemyDAO):
    def __init__(self, session=None, schema=None):
        self.session = session
        self.connection = session.connection
        self.schema = schema

    def get_query(self, query_def, style="cursor", **kwargs):

        # Initialize registries.
        source_registry = {'join_tree': {'children': {}}, 'nodes': {}}
        entity_registry = {}

        # Process 'from'.
        froms = []
        for source_def in query_def.get('FROM', []):
            if not source_def: continue
            # Process any joins the source has and add to from obj.
            source = self.add_joins(source_registry, source_def)
            froms.append(source)

        # Process 'select'.
        selections = []
        if isinstance(query_def.get('SELECT'), str):
            query_def['SELECT'] = [query_def['SELECT']]
        for entity_def in query_def.get('SELECT', []):
            if not entity_def: continue
            entity = self.get_registered_entity(
                source_registry, entity_registry, entity_def)
            selections.append(entity)

        # Process 'where'.
        # Where def is assumed to be a list with three parts:
        # entity, op, value.
        wheres = []
        for where_def in query_def.get('WHERE', []):
            if not where_def: continue

            # Get registered entity.
            entity = self.get_registered_entity(
                source_registry, entity_registry, where_def[0])

            # Handle mapped operators.
            if self.ops.has_key(where_def[1]):
                op = getattr(entity, self.ops[where_def[1]])
                where = op(where_def[2])
            # Handle all other operators.
            else:
                where = mapped_entity.op(where_def[1])(where_def[2])
            wheres.append(where)
            
        # Process 'group_by'.
        group_bys = []
        for entity_def in query_def.get('GROUP_BY', []):
            if not entity_def: continue

            entity_def = self.prepare_entity_def(entity_def)

            # If entity is a histogram entity, get histogram entities for grouping.
            if entity_def.get('AS_HISTOGRAM'):
                histogram_entity = self.get_histogram_entity(
                    source_registry, entity_registry, entity_def)
                group_bys.extend([histogram_entity])

            # Otherwise just use the plain entity for grouping.
            else:
                entity = self.get_registered_entity(source_registry, entity_registry, entity_def)
                group_bys.append(entity)

        # If 'select_group_by' is true, add group_by entities to select.
        if query_def.get('SELECT_GROUP_BY'):
            selections.extend(group_bys)

        # Process 'order_by'.
        order_bys = []
        for order_by_def in query_def.get('ORDER_BY', []):
            if not order_by_def: continue
            # If def is not a dict , we assume it represents an entity id.
            if not isinstance(order_by_def, dict):
                order_by_def = {'ENTITY': order_by_def}
            # Get registered entity.
            entity = self.get_registered_entity(
                source_registry, entity_registry, order_by_def['ENTITY'])

            # Assign direction.
            if order_by_def.get('DIRECTION') == 'desc':
                order_by_entity = desc(entity)
            else:
                order_by_entity = asc(entity)
            order_bys.append(order_by_entity)


        # Process joins.
        for node in source_registry['join_tree']['children'].values():
            join_chain = self.process_join_tree_node(node)
            if join_chain:
                join_point = join_chain[0]
                for target in join_chain[1:]:
                    join_point = orm.join(
                        join_point, target)
                froms.append(join_point)

        # Assemble query.
        q = self.session.query(*selections)\
                .select_from(*froms)\
                .filter(*wheres)\
                .group_by(*group_bys)\
                .order_by(*order_bys)

        # If style is cursor, decorate w/ cursor-style
        # fetch functions.
        if style == 'cursor':
            self.cursorify_query(q)

        return q

    def get_query_proxy(self, q):
        return QueryProxy(q)

    def get_registered_source(self, source_registry, source_def):
        source_def = self.prepare_source_def(source_def)

        node = source_registry['nodes'].get(source_def['ID'])
        if not node:
            # If 'source' is a dict , we assume it's a query object and process it.
            if isinstance(source_def['SOURCE'], dict):
                source = self.get_query(source_def['SOURCE']).alias(source_def['ID'])

            # Otherwise we process the source path...
            else:
                parts = source_def['SOURCE'].split('.')

                # Register dependencies and add to join tree.
                node = None
                parent_node = source_registry['join_tree']
                if len(parts) < 2:
                    source_id = '.'.join(parts)
                    source = aliased(self.schema['sources'][source_id])
                    node = {
                        'source': source,
                        'children': {}
                    }
                    source_registry['nodes'][source_id] = node
                    parent_node['children'][source_id] = node
                else:
                    for i in range(1, len(parts) + 1):
                        parent_id = '.'.join(parts[:i])
                        if source_registry['nodes'].has_key(parent_id):
                            node = source_registry['nodes'][parent_id]
                        else:
                            if i == 1:
                                source = aliased(self.schema['sources'][parent_id])
                                parent_attr = parent_id
                            else:
                                grandparent_id = '.'.join(parts[:i-1])
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

    def get_registered_entity(self, source_registry, entity_registry, entity_def):

        entity_def = self.prepare_entity_def(entity_def)

        # Map and register entity if not in the registry.
        if not entity_registry.has_key(entity_def['ID']):

            mapped_entities = {}

            # Replace entity tokens in expression w/ mapped entities.
            # This will be called for each token match.
            def replace_token_with_mapped_entity(m):
                token = m.group(1)
                parts = token.split('.')
                attr_id = parts[-1]
                source_def = '.'.join(parts[:-1])
                if source_def:
                    source = self.get_registered_source(
                        source_registry, source_def)
                    mapped_entities[token] = getattr(source, attr_id)
                else:
                    mapped_entities[token] = self.get_registered_source(
                        source_registry, attr_id)
                return "mapped_entities['%s']" % token

            entity_code = re.sub('{{(.*?)}}', replace_token_with_mapped_entity, entity_def['EXPRESSION'])

            # Evaluate and label.
            mapped_entity = eval(entity_code)
            if isinstance(mapped_entity, AliasedClass): 
                mapped_entity._sa_label_name = entity_def['ID']
            else:
                mapped_entity = mapped_entity.label(entity_def['ID'])

            # Register.
            entity_registry[entity_def['ID']] = mapped_entity

        return entity_registry[entity_def['ID']]

    def cursorify_query(self, q):
        def fetchall(self):
                return self.all()
        q.fetchall = types.MethodType(fetchall, q)

        def fetchone(self):
            return self.one()
        q.fetchone = types.MethodType(fetchone, q)
