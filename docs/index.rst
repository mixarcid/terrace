.. Terrace documentation master file, created by
   sphinx-quickstart on Wed May 10 11:03:28 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Terrace
===================================

Terrace is a library for writing concise and maintainable PyTorch code. There
are two main features Terrace introduces:

* :ref:`Modules` allow concise structuring of PyTorch models entirely in the
  model's ``forward`` method -- no more constantly switching between the
  ``__init__`` and ``forward`` methods when you make changes.
* :ref:`Batches` enable making nice object-oriented code compatible with PyTorch
  dataloaders. No more passing 10 tensors as arguments to your functions!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   guides/index
   api/terrace

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
