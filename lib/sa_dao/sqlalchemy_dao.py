from sa_dao.util import memoized
from sqlalchemy.sql import *
from sqlalchemy.sql import compiler
from sqlalchemy import cast, String, case
from sqlalchemy.sql.expression import join
from sqlalchemy.util import KeyedTuple
from sqlalchemy.engine import RowProxy
import sys
import re
import copy
import platform
import ast, _ast


class DefaultEntityExpressionASTVisitor(ast.NodeVisitor):
    """ Walks Entity Expression AST to validate nodes. 
    Used in DefaultEntityExpressionValidator below.
    """

    def __init__(self, 
                 valid_funcs=[],
                 name_validator=lambda name: False
                ):
        self.valid_funcs = valid_funcs
        self.name_validator = name_validator
        super(self.__class__, self).__init__()

    def visit(self, node):
        """ Visit method.  Only allowed statements are expressions.
        All others (imports, execs, etc. are verboten). """
        if isinstance(node, _ast.stmt) and not isinstance(node, _ast.Expr):
            raise Exception(("Invalid statement type '%s'.  Only 'Expr'"
                             " statements are allowed.") % type(node))
        else:
            super(self.__class__, self).visit(node)

    def validate_call_node(self, node):
        if not hasattr(node.func, 'value') or not hasattr(node.func, 'attr'):
            raise Exception("Invalid call: '%s'" % node.__dict__)
        func_name = "%s.%s" % (node.func.value.id, node.func.attr)
        if not func_name in self.valid_funcs:
            raise Exception("Invalid function: '%s.%s' is not in valid_funcs" % 
                            ( node.func.value.id, node.func.attr)
                           )

    def visit_Name(self, node):
        self.name_validator(node.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        self.validate_call_node(node)
        for arg in node.args or []: self.visit(arg)
        for kwarg in node.keywords or []: self.visit(kwarg)

class InvalidExpressionError(Exception): pass

class DefaultEntityExpressionValidator(object):
    """ Validates an entity expression via python's 
    Abstract Source Tree module.  This helps to guard against 
    sqlinjection, eval nastiness. """

    # Column tokens *MUST* have this format: __DATASOURCE__COLUMNNAME
    # e.g. __results__cell__id
    entity_re = '\b(__(\w+?))+\b'

    ast_visitor_class = DefaultEntityExpressionASTVisitor

    def __init__(self, valid_funcs=[]):

        self.ast_visitor = self.ast_visitor_class(
            name_validator=lambda name: re.match(self.entity_re, name),
            valid_funcs=valid_funcs
        )

    def validate_expression(self, expression):
        try:
            self.ast_visitor.visit(ast.parse(expression))
        except Exception, e:
            raise InvalidExpressionError("Expression '%s' is invalid: %s" % (
                expression,
                str(e)
            ))


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

    # Only these function calls are allowed in expressions.
    valid_funcs = [
        'func.sum',
        'func.min',
        'func.max',
    ]

    expression_validator_class = DefaultEntityExpressionValidator

    def __init__(self, connection=None, schema=None, expression_locals={}):
        self.connection = connection
        self.schema = schema
        self.expression_validator = self.expression_validator_class(
            valid_funcs=self.valid_funcs)
        self.expression_locals = expression_locals

        # memoize get_query_results to save on queries.
        self.get_query_results = memoized(self.get_query_results)

    def create_all(self, **kwargs):
        self.schema['metadata'].create_all(bind=self.connection, **kwargs)

    def drop_all(self):
        self.schema['metadata'].drop_all(bind=self.connection, **kwargs)

    def join_(self, *args, **kwargs):
        return join(*args, **kwargs)

    # Given a list of query definitions, return results.
    def execute_queries(self, query_defs=[], **kwargs):
        results = {}
        for query_def in query_defs:
            q = self.get_query(query_def, **kwargs)

            # Compile to raw sql first, to allow for memoization.
            # Also avoids Jython zxJDBC issues w/ histogram entities.
            q_sql = self.query_to_raw_sql(q)

            # By default, return results as dictionaries.
            q_results = self.get_query_results(
                q_sql, query_def.get('AS_DICTS', True))

            results[query_def['ID']] = q_results
        return results

    def get_result_cursor(self, q):
        return self.connection.execute(q)

    def get_query_results(self, q, as_dicts):
        rows = self.get_result_cursor(q).fetchall()
        if as_dicts:
            results = []
            for row in rows:
                if not isinstance(row, KeyedTuple) \
                   and not isinstance(row, RowProxy):
                    results.append({'obj': row})
                else:
                    results.append(dict(zip(row.keys(), row)))
        else:
            results = [r for r in rows]
        return results

    def generate_source_registry(self):
        return {'join_tree': {'children': {}}, 'nodes': {}}

    # Return a query object for the given query definition. 
    def get_query(self, query_def, return_registries=False, **kwargs):

        # Initialize registries if not provided.
        kwargs.setdefault('source_registry', self.generate_source_registry())
        kwargs.setdefault('entity_registry', {})
        kwargs.setdefault('token_registry', {})

        # Convert simple select to query def.
        if isinstance(query_def, str):
            query_def = {'SELECT': query_def}

        # Fetch registries from def, or initialize registries.
        source_registry = query_def.get(
            'source_registry', {'join_tree': {'children': {}}, 'nodes': {}})
        entity_registry = query_def.get('entity_registry', {})

        # Process 'from'.
        for source_def in query_def.get('FROM', []):
            if not source_def: continue
            # Process any joins the source has and add to source registry.
            self.add_source(source_def=source_def, **kwargs)

        # Process 'select'.
        selections = []
        if isinstance(query_def.get('SELECT'), str):
            query_def['SELECT'] = [query_def['SELECT']]
        for entity_def in query_def.get('SELECT', []):
            if not entity_def: continue
            entity = self.get_registered_entity(entity_def=entity_def, 
                                                **kwargs)
            selections.append(entity)

        # Process 'where'.
        # Where def is assumed to be a list with three parts:
        # entity, op, value.
        wheres = []
        for where_def in query_def.get('WHERE', []):
            if not where_def: continue
            where = self.process_where_def(where_def=where_def, **kwargs)
            wheres.append(where)
            
        # Process 'group_by'.
        group_bys = []
        for entity_def in query_def.get('GROUP_BY', []):
            if not entity_def: continue
            entity = self.get_registered_entity(entity_def=entity_def, 
                                                **kwargs)
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
                entity_def=order_by_def['ENTITY'], **kwargs)

            # Assign direction.
            if order_by_def.get('DIRECTION') == 'desc':
                order_by_entity = desc(entity)
            else:
                order_by_entity = asc(entity)
            order_bys.append(order_by_entity)

        # Process joins.
        froms = self.process_sources(kwargs.get('source_registry'))

        # Assemble query.
        q = self.assemble_query(
            selections=selections,
            froms=froms,
            wheres=wheres,
            group_bys=group_bys,
            order_bys=order_bys
        )

        if not return_registries:
            return q
        else:
            return q, {
                'entities': kwargs.get('entity_registry'), 
                'sources': kwargs.get('source_registry'),
                'tokens': kwags.get('token_registry'),
            }

    def assemble_query(self, selections=[], froms=[], wheres=[], group_bys=[],
                       order_bys=[], **kwargs):

        if len(wheres) > 0:
            whereclause = and_(*wheres)
        else:
            whereclause = None

        q = select(
            columns=selections,
            from_obj=froms,
            whereclause=whereclause,
            group_by=group_bys,
            order_by=order_bys,
            use_labels=True
        )
        return q

    def process_sources(self, source_registry=None, **kwargs):
        froms = []
        for node in source_registry['join_tree']['children'].values():
            join_chain = self.process_join_tree_node(node)
            if join_chain:
                join_point = join_chain[0]['source']
                for target in join_chain[1:]:
                    join_point = self.join_(join_point,
                                            target['source']['source'],
                                            **target['kwargs'])
                froms.append(join_point)
        return froms 

    def process_join_tree_node(self, node):
        join_chain = [node]
        for child_node in node.get('children', {}).values():
            join_chain.extend(self.process_join_tree_node(child_node))
        return join_chain

    # Prepare a source definition for use.
    def prepare_source_def(self, source_def):
        if not isinstance(source_def, dict):
            source_def = {'SOURCE': source_def}
        source_def.setdefault('ID', source_def['SOURCE'])

        return source_def

    # Get or register a source in a source registry.
    def get_registered_source(self, source_registry=None, source_def=None, 
                              **kwargs):
        #@TODO: perhaps consolidate join logic here w/
        # 'process_source'? we just add 'JOIN' defs here?
        source_def = self.prepare_source_def(source_def)

        node = source_registry['nodes'].get(source_def['ID'])
        if not node:
            # Register dependencies and add to join tree.
            parent_node = source_registry['join_tree']

            # If 'source' is a dict , we assume it's a query object and process it.
            if isinstance(source_def['SOURCE'], dict):
                source = self.get_query(
                    source_def['SOURCE'], 
                    token_registry=kwargs.get('token_registry')
                ).alias(source_def['ID'])
                node = {
                    'source': source,
                    'children': {}
                }
                source_registry['nodes'][source_def['ID']] = node
                parent_node['children'][source_def['ID']] = node
            # Otherwise we process the source path...
            else:
                parts = source_def['SOURCE'].split('__')
                for i in range(1, len(parts) + 1):
                    parent_id = '__'.join(parts[:i])
                    node = source_registry['nodes'].get(parent_id)
                    if not node:
                        parent_attr = parts[i-1]
                        source = self.schema['sources'][parent_attr]
                        node = {
                            'source': source,
                            'children': {}
                        }
                        source_registry['nodes'][parent_id] = node
                        parent_node['children'][parent_attr] = node

                    parent_node = node

        return node['source']

    def add_source(self, source_registry=None, entity_registry=None, 
                   source_def=None, **kwargs):
        """ Add joins to a given source. """
        source_def = self.prepare_source_def(source_def)
        source = self.get_registered_source(source_registry, source_def,
                                            **kwargs)
        source_node = source_registry['nodes'][source_def['ID']]

        for join_def in source_def.get('JOINS', []):
            if not isinstance(join_def, list):
                join_def = [join_def]
            if len(join_def) > 1:
                where_def = join_def[1]
                onclause = self.process_where_def(
                    where_def=where_def, 
                    source_registry=source_registry,
                    entity_registry=entity_registry,
                    **kwargs)
            else:
                onclause = None

            target_def = self.prepare_source_def(join_def[0])
            # Register target if not yet registered.
            self.get_registered_source(source_registry, target_def)
            target_node = source_registry['nodes'].get(target_def['ID'])

            # Add to child nodes of join target.
            target_node['children'][source_def['ID']] = {
                'source': source_node,
                'kwargs': {'onclause': onclause}
            }

        return source

    def process_where_def(self, where_def=None, **kwargs):
        left = self.process_where_element(
            element=where_def[0],
            **kwargs
        )
        right = self.process_where_element(
            element=where_def[2],
            **kwargs
        )
        if self.ops.has_key(where_def[1]):
            op = getattr(left, self.ops[where_def[1]])
            where = op(right)
        else:
            where = left.op(where_def[1])(right)
        return where

    def process_where_element(self, element=None, **kwargs):
        if isinstance(element, dict) and element.get('TYPE') == 'ENTITY':
            return self.get_registered_entity(entity_def=element, **kwargs)
        else:
            return element

    def prepare_entity_def(self, entity_def):
        # If item is not a dict, we assume it's a string-like object representing an entity expression.
        if not isinstance(entity_def, dict):
            entity_def = {'EXPRESSION': entity_def}
        # If item has no ID, assign an arbitrary id.
        entity_def.setdefault('ID', str(id(entity_def)))
        return entity_def


    # Get or register an entity.
    def get_registered_entity(self, source_registry={}, entity_registry={}, 
                              token_registry={}, entity_def=None, **kwargs):

        entity_def = self.prepare_entity_def(entity_def)

        # Map and register entity if not in the registry.
        if not entity_registry.has_key(entity_def['ID']):
            # Handle histogram entities.
            if entity_def.get('AS_HISTOGRAM'):
                mapped_entity = self.get_histogram_entity(
                    source_registry=source_registry, 
                    entity_registry=entity_registry,
                    entity_def=entity_def,
                    token_registry=token_registry,
                    **kwargs
                )
            # All other entities...
            else:
                # First validate the expression.  This will throw an error
                # if the expression is invalid.
                self.expression_validator.validate_expression(entity_def['EXPRESSION'])

                mapped_entities = {}

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
                            mapped_entities[token] = self.alter_col(
                                source.c.get(attr_id))
                    else:
                        mapped_entities[token] = self.get_registered_source(
                            source_registry, attr_id)
                    return "mapped_entities['%s']" % token

                expression_code = re.sub(
                    r'\b(__(\w+))+\b', replace_token_with_mapped_entity, 
                    entity_def['EXPRESSION'])

                # Evaluate.
                mapped_entity = self.eval_expression_code(
                    expression_code, globals(), locals()
                )

            # Label the entity.
            mapped_entity = mapped_entity.label(entity_def['ID'])

            # Register.
            entity_registry[entity_def['ID']] = mapped_entity

        return entity_registry[entity_def['ID']]

    def alter_col(self, col):
        """ Alter a column element. Intended to be subclassed. """
        return col
    
    def get_keyed_results(self, key_def=None, query_defs=None):

        # Initialize keyed results.
        keyed_results = {}

        # Shortcut for key entity.
        key_entity = key_def['KEY_ENTITY']
        key_entity = self.prepare_entity_def(key_entity)

        # If there was no label entity, use the key entity as the label entity.
        label_entity = key_def.setdefault('LABEL_ENTITY', copy.deepcopy(key_entity))
        label_entity = self.prepare_entity_def(label_entity)

        # Initialize token registry with key entities.
        token_registry = {
            'KEY': key_entity,
            'LABEL': label_entity,
        }

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
                key_query_def = key_def.get('QUERY')
                keys_labels = self.execute_queries(
                    query_defs=[key_def.get('QUERY')],
                    token_registry=token_registry).values()[0]

            # Pre-seed keyed results with keys and labels.
            for key_label in keys_labels:
                key = key_label[key_id]
                label = key_label[label_id]

                keyed_results[key] = {
                        "key": key,
                        "label": label,
                        "data": {}
                        }

        # Modify query defs to return as dicts.
        for query_def in query_defs:
            query_def["AS_DICTS"] = True

        # Execute primary queries.
        results = self.execute_queries(query_defs, token_registry=token_registry)

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
        if not dialect:
            dialect = self.get_dialect()
        compiler = q._compiler(dialect)
        class LiteralCompiler(compiler.__class__):
            def visit_bindparam(
                self, bindparam, within_columns_clause=False, 
                literal_binds=False, **kwargs
            ):
                return super(LiteralCompiler, self).render_literal_bindparam(
                    bindparam, within_columns_clause=within_columns_clause,
                    literal_binds=literal_binds, **kwargs
                )

        return LiteralCompiler(dialect, q).process(q)

    def get_histogram_classes(self, entity_def=None, **kwargs):
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
    def get_histogram_entity(self, source_registry=None, 
                             entity_registry=None, entity_def=None, **kwargs):
        entity_def = self.prepare_entity_def(entity_def)

        # Get or register base entity.
        # We copy entity_def and take out 'as_histogram' to avoid recursion.
        base_entity_def = {}
        base_entity_def.update(entity_def)
        base_entity_def['AS_HISTOGRAM'] = False
        base_entity = self.get_registered_entity(
            source_registry=source_registry,
            entity_registry=entity_registry, 
            entity_def=base_entity_def,
            **kwargs
        )

        # Get histogram classes.
        classes = self.get_histogram_classes(entity_def)

        # Assemble cases for case statement.
        cases = []
        for c in classes:
            case_obj = (
                    and_(base_entity >= c[0], base_entity < c[1]), 
                    self.get_histogram_class_label(c)
                    )
            cases.append(case_obj)

        # Return mapped entity.
        histogram_entity = case(cases)
        return histogram_entity

    def get_dialect(self, dialect_name=None):
        if not dialect_name:
            # If using jython w/ zxjdbc, need to get normal dialect
            # for bind parameter substitution.
            drivername = self.connection.engine.url.drivername
            m = re.match("(.*)\+zxjdbc", drivername)
            if m:
                dialect_name = self.get_dialect(m.group(1))
            # Otherwise use the normal session dialect.
            else:
                return self.connection.dialect

        dialects_module = __import__("sqlalchemy.dialects", fromlist=[dialect_name])
        return getattr(dialects_module, dialect).dialect()

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

    def eval_expression_code(self, expression_code, globals_, locals_):
        merged_locals = {}
        merged_locals.update(self.expression_locals)
        merged_locals.update(locals_)
        return eval(expression_code, globals_, merged_locals)

