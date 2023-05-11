.. _Batches:

Batches
========

.. jupyter-execute::
    :hide-code:
    
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join('..', 'src')))

Terrace ``Batches`` are a way to object-orientify PyTorch code. Using vanilla
PyTorch ``DataLoader``, your dataset can return tensors (and tuples and dicts of tensors),
but not arbitrary classes -- PyTorch doesn't know how to collate them. Not anymore!

Let's suppose you're dealing with a dataset of people. For each person, there is
an image of their face and a string name. In the dark ages, you would have loaded
the face and name into two seperate tensors and passed them individually as arguments
to your functions. Now, however, you can make a ``Person`` class.

.. jupyter-execute::

    import torch
    import terrace as ter

    MAX_NAME_LEN = 128
    IMG_SIZE = 256
    class Person(ter.Batchable):
        
        face: torch.Tensor
        name: torch.Tensor
        
        def __init__(self): # fake person data
            self.face = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            self.name = torch.zeros((MAX_NAME_LEN,))

Since we've inherited from the ``Batchable`` class, terrace knows how to collate
multiple people into a single ``Batch[Person]``.

.. jupyter-execute::

    dave = Person()
    rhonda = Person()
    batch = ter.collate([dave, rhonda])
    print(batch)

    # Batches have a length and can be indexed, just like lists
    print(len(batch))
    
    # But you can also access their (batched) members, just like
    # objects of the original class
    print(dave.name.shape)
    print(batch.name.shape) # notice the extra batch dimension
    
    print(batch[0]) # un-batchification

.. note::

    Notice that last line -- when we index a batch, it returns a ``BatchView[Person]``,
    not a ``Person`` directly. This is for performance reasons -- under the hood,
    a ``BatchView`` lazily indexes the members of its parent batch. The vast majority
    of the time, this ``BatchView`` will act exactly like a ``Person``. If it doesn't,
    please submit a `bug report <https://github.com/mixarcid/terrace/issues/new>`_.

Now we can use Terrace's ``DataLoader`` to automatically batchify a 
custom dataset.

.. jupyter-execute::

    class PersonDataset(torch.utils.data.Dataset):
        
        def __len__(self):
            return 16
        
        def __getitem__(self, index):
            return Person()
        
    batch_size = 8
    dataset = PersonDataset()
    loader = ter.DataLoader(dataset, batch_size=batch_size)
    for batch in loader:
        print(batch)

Additionally, we can create new batches of people directly (e.g. in a generative
model).

.. jupyter-execute::

    batch = ter.Batch(Person,
                      face=torch.zeros((batch_size, 3, IMG_SIZE, IMG_SIZE)),
                      name=torch.zeros((batch_size, MAX_NAME_LEN)))
    print(batch)

.. warning::

    Creating a batch directly from batched member data is dangerous because 
    Terrace (currently) doesn't do any checking to make sure you've input 
    reasonable arguments.


+++++++++++++++
Graphs
+++++++++++++++

In addition to batchifying everyday data, Terrace has special graph functionality.
With ``GraphBatches``, terraces provides an higher-level object-oriented abstraction
over `DGL <https://www.dgl.ai/>`_ graphs. (In the future, `PyG <https://pyg.org/>`_
might be added as a backend as well.).

You can create ``Batchable`` subclasses for both node and edge data. Here's how.

.. jupyter-execute::

    class Atom(ter.Batchable):

        # let's suppose atoms have a 3D position
        # and an atomic mass

        position: torch.Tensor
        mass: torch.Tensor

        def __init__(self):
            """ Fill with dummy data """
            self.position = torch.zeros((3,))
            self.mass = torch.zeros((1,))

    class Bond(ter.Batchable):

        order: torch.Tensor

        def __init__(self):
            self.order = torch.zeros((1,))

    # to create a Terrace graph, we need the node data, edge indexes,
    # and (optionally) edge data
    ndata = [ Atom(), Atom(), Atom() ]
    edges = [ (0, 1), (0, 2)]
    edata = [ Bond(), Bond() ]

    mol = ter.Graph(ndata, edges, edata)
    print(mol)

    # we can access their node and edge data batches with ndata and edata
    print(mol.ndata)
    print(mol.edata)

    # If we want, we can also get the underlying DGL graph.
    # This is necessary when we want to create wrapper modules
    # for the DGL model classes
    print(mol.dgl())

Of course, we can combine graph batches and regular batches into arbitrarily
complex nested structures.

.. jupyter-execute::

    class MolAndData(ter.Batchable):

        # all Batchable classes are dataclasses,
        # so we don't actually need a constructor

        mol: ter.Graph[Atom, Bond]
        data: torch.Tensor

    data = torch.zeros((8,))
    mol_and_data = MolAndData(mol, data)
    batch = ter.collate([ mol_and_data, mol_and_data, mol_and_data])
    print(batch)

This is a very simple example, but feel free to go wild.

+++++++++++++++++++
Advanced features
+++++++++++++++++++

To enable your code to be even more object-oriented, Terrace allows you
to define member functions for your batches. If you define a member
function in your ``Batchable`` class with the name ``batch_{func_name}``,
batches of your class will all have the member function ``{func_name}``.
Here's how we can modify the ``Person`` class from above to use this feature.

.. jupyter-execute::

    class Person(ter.Batchable):
    
        face: torch.Tensor
        name: torch.Tensor

        def say_hi(self):
            print("Hello, I'm a person")

        def batch_say_hi(self):
            print(f"Hello, I'm a batch of {len(self)} people")
        
        def __init__(self): # fake person data
            self.face = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            self.name = torch.zeros((MAX_NAME_LEN,))

    person = Person()
    batch = ter.collate([Person(), Person(), Person()])

    person.say_hi()
    batch.say_hi()
