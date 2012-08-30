from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import (Table, Column, ForeignKey, ForeignKeyConstraint, 
                        Integer, String, Float)


def setUpSchemaAndData1(session):
    schema = {}
    classes = {}

    Base = declarative_base()

    # TestClass1
    class TestClass1(Base):
        __table__name = 'testclass1'
        id = Column(Integer, primary_key=True)
        name = Column(String)

        def __init__(self, id=None, name=None):
            self.id = id
            self.name = name

    classes['TestClass1'] = TestClass1

    # TestClass2
    class TestClass2(Base):
        __table__name = 'testclass2'
        id = Column(Integer, primary_key=True)
        name = Column(String)
        child_id = Column(Integer, ForeignKey(TestClass1.__table__.c.id))

        children = relationship(TestClass1)

        def __init__(self, id=None, name=None, children=[]):
            self.id = id
            self.name = name
            self.children = children

    classes['TestClass2'] = TestClass2

    # Save classes to schema.
    schema['classes'] = classes

    # Setup tables.
    Base.metadata.create_all(session.bind)

    # Generate data.
    tc1s = []
    tc2s = []
    for i in range(5):
        tc1 = TestClass1(
            id=i,
            name="tc1_%s" % i
        )
        tc1s.append(tc1)
        session.add(tc1)

        tc2 = TestClass2(
            id=i,
            name="tc2_%s" % i
        )
        tc2s.append(tc2)
        session.add(tc2)
    session.commit()

    for i in range(len(tc2s)):
        tc2 = tc2s[i]
        child_tc1s = [tc1s[i], tc1s[ (i + 1) % len(tc1s)]]
        tc2.children = child_tc1s
        session.add(tc2)
    session.commit()

    return schema
