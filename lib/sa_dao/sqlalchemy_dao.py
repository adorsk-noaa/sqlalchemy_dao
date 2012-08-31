from sqlalchemy.sql import *
from sqlalchemy.sql import compiler
from sqlalchemy import cast, String, case
from sqlalchemy.sql.expression import join
from sqlalchemy.util._collections import NamedTuple
import sys
import re
import copy
import platform


class SqlAlchemyDAO(object):

    ops = {
            '==': '__eq__',
            '!=': '__ne__',
            '<': '__lt__',
            '>': '__gt__',
            '<=': '__le__',
            '>=': '__ge__',
            'in': 'in_',
            }

    def __init__(self, connection=None, schema=None):
        self.connection = connection
        self.schema = schema

    # Given a list of query definitions, return results.
    def execute_queries(self, query_defs=[]):
        results = {}
        for query_def in query_defs:
            q = self.get_query(query_def)
            #print "q is: ", self.query_to_raw_sql(q)
            # If using jython, compile first.  Sometimes
            # there are issues w/ using histograms.
            if platform.system() == 'Java':
                q = self.query_to_raw_sql(q)
            query_proxy = self.get_query_proxy(q)
            rows = query_proxy.fetchall()
            # By default, return results as dictionaries.
            if query_def.get('AS_DICTS', True):
                q_results = []
                for row in rows:
                    if not isinstance(row, NamedTuple):
                        q_results.append({'obj': row})
                    else:
                        q_results.append(dict(zip(row.keys(), row)))
            else:
                q_results = rows
            results[query_def['ID']] = q_results
        return results
        

    # Return a query object for the given query definition. 
    def get_query(self, query_def, **kwargs):

        # Initialize registries.
        source_registry = {'join_tree': {'children': {}}, 'nodes': {}}
        entity_registry = {}

        # Process 'from'.
        from_obj = []
        for source_def in query_def.get('FROM', []):
            if not source_def: continue
            # Process any joins the source has and add to from obj.
            source = self.add_joins(source_registry, source_def)
            from_obj.append(source)

        # Process 'select'.
        columns = []
        for entity_def in query_def.get('SELECT', []):
            if not entity_def: continue
            entity = self.get_registered_entity(source_registry, entity_registry, entity_def)
            columns.append(entity)

        # Process 'where'.
        # Where def is assumed to be a list with three parts:
        # entity, op, value.
        wheres = []
        for where_def in query_def.get('WHERE', []):
            if not where_def: continue

            # Get registered entity.
            entity = self.get_registered_entity(source_registry, entity_registry, where_def[0])

            # Handle mapped operators.
            if self.ops.has_key(where_def[1]):
                op = getattr(entity, self.ops[where_def[1]])
                where = op(where_def[2])
            # Handle all other operators.
            else:
                where = mapped_entity.op(where_def[1])(where_def[2])
            wheres.append(where)
        # Combine wheres into one clause.
        whereclause = None
        if len(wheres) > 0:
            whereclause = and_(*wheres)
            
        # Process 'group_by'.
        group_by = []
        for entity_def in query_def.get('GROUP_BY', []):
            if not entity_def: continue

            entity_def = self.prepare_entity_def(entity_def)

            # If entity is a histogram entity, get histogram entities for grouping.
            if entity_def.get('AS_HISTOGRAM'):
                histogram_entity = self.get_histogram_entity(source_registry, entity_registry, entity_def)
                group_by.extend([histogram_entity])

            # Otherwise just use the plain entity for grouping.
            else:
                entity = self.get_registered_entity(source_registry, entity_registry, entity_def)
                group_by.append(entity)

        # Process 'order_by'.
        order_by = []
        for order_by_def in query_def.get('ORDER_BY', []):
            if not order_by_def: continue
            # If def is not a dict , we assume it represents an entity id.
            if not isinstance(order_by_def, dict):
                order_by_def = {'ENTITY': order_by_def}
            # Get registered entity.
            entity = self.get_registered_entity(source_registry, entity_registry, order_by_def['ENTITY'])

            # Assign direction.
            if order_by_def.get('DIRECTION') == 'desc':
                order_by_entity = desc(entity)
            else:
                order_by_entity = asc(entity)

            order_by.append(order_by_entity)

        # If 'select_group_by' is true, add group_by entities to select.
        if query_def.get('SELECT_GROUP_BY'):
            columns.extend(group_by)


        # Process joins and add them to the from obj.
        for node_id, node in source_registry['join_tree']['children'].items():
            joins = self.process_join_tree(source_registry['join_tree'])
            if joins:
                # Go from child to parent.
                joins.reverse()
                source = joins[0]
                for s in joins[1:]:
                    source = join(source, s)
                from_obj.append(source)

        # Return the query object.
        q = select(
                columns=columns, 
                from_obj=from_obj,
                whereclause=whereclause,
                group_by=group_by,
                order_by=order_by,
                use_labels=True
                )
        return q

    def process_join_tree_node(self, node):
        join_chain = [node['source']]
        for child_node in node['children'].values():
            join_chain.extend(self.process_join_tree_node(child_node))
        return join_chain

    # Prepare a source definition for use.
    def prepare_source_def(self, source_def):
        if not isinstance(source_def, dict):
            source_def = {'SOURCE': source_def}
        source_def.setdefault('ID', source_def['SOURCE'])

        return source_def

    # Get or register a source in a source registry.
    def get_registered_source(self, source_registry, source_def):
        source_def = self.prepare_source_def(source_def)
        
        # Process source if it's not in the registry.
        if not source_registry['nodes'].has_key(source_def['ID']):

            # If 'source' is a dict , we assume it's a query object and process it.
            if isinstance(source_def['SOURCE'], dict):
                source = self.get_query(source_def['SOURCE']).alias(source_def['ID'])
            # Otherwise we process the source path...
            else:
                parts = source_def['SOURCE'].split('.')

                # The source is the last part of the path.
                source = self.schema['sources'][parts[-1]]

                # Save the path to the join tree.
                parent = source_registry['join_tree']
                for part in parts:
                    if not parent['children'].has_key(part):
                        parent['children'][part] = {
                                'source': self.schema['sources'][part],
                                'children': {}
                                }
                    parent = parent['children'][part]

            # Save the aliased source to the registry.
            source_registry['nodes'][source_def['ID']] = source

        return source_registry['nodes'][source_def['ID']]



    # Add joins to source.
    def add_joins(self, source_registry, source_def):
        source_def = self.prepare_source_def(source_def)

        # Get or register the source.
        source = self.get_registered_source(source_registry, source_def)

        # Recursively process joins.
        for join_def in source_def.get('JOINS', []):
            # Convert to list if given as a non-list.
            if not isinstance(join_def, list):
                join_def = [join_def]

            # Get onclause if given.
            if len(join_def) > 1:
                onclause = join_def[1]
            else:
                onclause = None

            source = source.join(self.add_joins(source_registry, join_def[0]), onclause=onclause)

        return source

    def prepare_entity_def(self, entity_def):
        # If item is not a dict, we assume it's a string-like object representing an entity expression.
        if not isinstance(entity_def, dict):
            entity_def = {'EXPRESSION': entity_def}
        # If item has no ID, assign an arbitrary id.
        entity_def.setdefault('ID', str(id(entity_def)))
        return entity_def


    # Get or register an entity.
    def get_registered_entity(self, source_registry, entity_registry, entity_def):

        entity_def = self.prepare_entity_def(entity_def)

        # Map and register entity if not in the registry.
        if not entity_registry.has_key(entity_def['ID']):

            mapped_entities = {}

            # Replace entity tokens in expression w/ mapped entities.
            # This will be called for each token match.
            def replace_token_with_mapped_entity(m):
                token = m.group(1)
                m = re.match('(.*)\.(.*)', token)
                if m:
                    source_def = m.group(1)
                    column_id = m.group(2)
                    source = self.get_registered_source(source_registry, source_def)
                    mapped_entities[token] = source.c[column_id]
                    return "mapped_entities['%s']" % token

            entity_code = re.sub('{{(.*?)}}', replace_token_with_mapped_entity, entity_def['EXPRESSION'])

            # Evaluate and label.
            mapped_entity = eval(entity_code)
            mapped_entity = mapped_entity.label(entity_def['ID'])

            # Register.
            entity_registry[entity_def['ID']] = mapped_entity

        return entity_registry[entity_def['ID']]

    
    def get_keyed_results(self, key_def=None, query_defs=None):

        # Initialize keyed results.
        keyed_results = {}

        # Shortcut for key entity.
        key_entity = key_def['KEY_ENTITY']
        key_entity = self.prepare_entity_def(key_entity)

        # If there was no label entity, use the key entity as the label entity.
        label_entity = key_def.setdefault('LABEL_ENTITY', copy.deepcopy(key_entity))
        label_entity = self.prepare_entity_def(label_entity)

        # Shortcuts to key and label ids.
        key_id = key_entity['ID']
        label_id = label_entity['ID']

        # If all values should be selected for the key entity...
        if key_entity.get('ALL_VALUES'):

            # If key entity is histogram, then generate the keys and labels.
            if key_entity.get('AS_HISTOGRAM'):
                keys_labels = []
                classes = self.get_histogram_classes(key_entity)
                for c in classes:
                    keys_labels.append({
                        key_entity['ID']: self.get_histogram_class_label(c)
                        })

            # Otherwise select the keys and labels per the key_def...
            else:
                # Select keys and labels.
                # We merge the key query attributes with our overrides.
                keys_labels = self.execute_queries(
                    query_defs = [
                        dict(key_def.items() + {
                            'ID': 'keylabel_q', 
                            'AS_DICTS': True, 
                            'GROUP_BY': [key_entity, label_entity],
                            'SELECT_GROUP_BY': True
                            }.items() )
                        ]).values()[0]

            # Pre-seed keyed results with keys and labels.
            for key_label in keys_labels:
                key = key_label[key_id]
                label = key_label[label_id]

                keyed_results[key] = {
                        "key": key,
                        "label": label,
                        "data": {}
                        }

        # Modify query defs.
        for query_def in query_defs:
            query_def["AS_DICTS"] = True

        # Execute primary queries.
        results = self.execute_queries(query_defs)

        # For each result set...
        for result_set_id, result_set in results.items():
            # For each result in the result set...
            for result in result_set:

                # Get the result's key.
                result_key = result.get(key_id)

                # If there was a key...
                if result_key:
                    # Get the label.
                    result_label = result.get(label_id)
                    # Get or create the keyed result.
                    keyed_result = keyed_results.setdefault(result_key, {
                        "key": result_key,
                        "label": result_label,
                        "data": {}
                        })

                    # Add the result to the keyed_result data.
                    keyed_result['data'][result_set_id] = result

        # Return the keyed results.
        return keyed_results.values()
                

    # Get raw sql for given query parameters.
    def get_sql(self, query_def, dialect=None, **kwargs):
        q = self.get_query(query_def, **kwargs)
        return self.query_to_raw_sql(q, dialect=dialect)

    # Compile a query into raw sql.
    def query_to_raw_sql(self, q, dialect=None):

        # Get dialect object.
        if not dialect:
            # If using jython w/ zxjdbc, need to get normal dialect
            # for bind parameter substitution.
            drivername = self.connection.engine.url.drivername
            m = re.match("(.*)\+zxjdbc", drivername)
            if m:
                dialect = self.get_dialect(m.group(1))
            # Otherwise use the normal session dialect.
            else:
                dialect = self.connection.dialect
        else:
            dialect = self.get_dialect(dialect)

        comp = compiler.SQLCompiler(dialect, q)
        enc = dialect.encoding
        params = {}
        for k,v in comp.params.iteritems():
            if isinstance(v, unicode):
                v = v.encode(enc)
            if isinstance(v, str):
                v = comp.render_literal_value(v, str)
            params[k] = v
        raw_sql = (comp.string.encode(enc) % params).decode(enc)
        return raw_sql

    def get_histogram_classes(self, entity_def):
        # Return classes if they were provided.
        if entity_def.get('CLASSES'):
            classes = entity_def['CLASSES']
        # Otherwise generate classes from min/max, num_classes.
        else:
            # Get min/max if not provided, via the 'context'.
            if entity_def.get('MIN') == None or entity_def.get('MAX') == None \
            or entity_def.get('MINAUTO') or entity_def.get('MAXAUTO'):
                SELECT = []
                for m in ['MIN', 'MAX']:
                    minmax_entity_def = {'ID': m, 'EXPRESSION': "func.%s(%s)" % (m.lower(), entity_def.get('EXPRESSION'))}
                    SELECT.append(minmax_entity_def)

                # We merge the context query attributes with our overrides.
                minmax = self.execute_queries(query_defs=[
                    dict(entity_def.get('CONTEXT', {}).items() + {
                        'ID': 'stats_q', 
                        'AS_DICTS': True, 
                        'SELECT': SELECT}.items()
                        ) 
                    ]).values()[0][0]

                # set MIN and MAX only if not provided, or if *AUTO is true.
                for m in ['MIN', 'MAX']:
                    # Set to 0 if None.
                    minmax.setdefault(m, 0)

                    # If auto, use minmax result.
                    if entity_def.get("%sAUTO" % m):
                        entity_def[m] = minmax[m]
                    # Otherwise, only set if not already set.
                    else:
                        entity_def.setdefault(m, minmax[m])

            entity_min = entity_def['MIN']
            entity_max = entity_def['MAX']
            num_classes = entity_def.get('NUM_CLASSES', 10)
            class_width = (entity_max - entity_min)/float(num_classes)
            
            # Create classes.
            classes = []
            for n in range(num_classes):
                cmin = entity_min + n * class_width
                cmax = cmin + class_width
                classes.append([cmin, cmax])

        return classes

    def get_histogram_class_label(self, c):
        class_label = "[%s, %s)" % (c[0], c[1])
        return class_label

    # Get histogram entities for a given entity.
    def get_histogram_entity(self, source_registry, entity_registry, entity_def):
        entity_def = self.prepare_entity_def(entity_def)

        # Get or register entity.
        entity = self.get_registered_entity(source_registry, entity_registry, entity_def)

        # Get histogram classes.
        classes = self.get_histogram_classes(entity_def)

        # Assemble cases for case statement.
        cases = []
        for c in classes:
            case_obj = (
                    and_(entity >= c[0], entity < c[1]), 
                    self.get_histogram_class_label(c)
                    )
            cases.append(case_obj)

        # Get labeled entity.
        histogram_entity = case(cases).label(entity_def['ID'])

        return histogram_entity

    def get_dialect(self, dialect):
        try:
            dialects_module = __import__("sqlalchemy.dialects", fromlist=[dialect])
            return getattr(dialects_module, dialect).dialect()
        except:
            return None

    def get_connection_parameters(self):
        engine = self.connection.engine
        connection_parameters = {}
        parameter_names = [
                "drivername",
                "host",
                "database",
                "username",
                "password",
                "port"
                ]
        for parameter in parameter_names:
            connection_parameters[parameter] = getattr(engine.url, parameter)

        return connection_parameters

    def query(self, query_def):
        return self.get_query(query_def=query_def)

    def save(self, obj, commit=True):
        self.session.add(obj)
        if commit:
            self.commit()

    def commit(self):
        self.session.commit()
