RELEASE NOTES FOR graphlib and gnnlib
(c) Edwin Heredia, 2016


1) Running nosetests requires installation of and nose
The conventional way to install nose is: 
	sudo easy_install pip
	sudo pip install nose

However in a recent copy of Mac OS X (El Capitan) the above procedure failed. Pip 
was unable to install nose. The following command was able to install nose: 
	sudo easy_install nose

2) Better statistics using the coverage module
Nose tests can display code coverage information using the coverage module. In a Mac OS X the install
procedure is:
    sudo easy_install coverage

3) Invoking nose with coverage:
	3.1 The folder that contains the source code also includes test files (test_*.py) with unit tests for
		the graph and gnn libraries. In order to run the unit tests launch a terminal window and
	    and go to the folder that contains source code and test files. Then type the following command:

	        nosetests --with-coverage -vv

	    This command runs all unit tests (for both libraries) and reports current coverage.

	    If you would like to run only one set of tests (either for gnnlib or for graphlib) then specify
	    the file name in the nosetests command:

	        nosetests --with-coverage -vv test_gnn.py

	3.2 The tests can also run from an IDE like PyCharm. After loading the project, right-click on the
	    test file (in the project explorer view) and select the "Run Unittests" option

4) As of 4/20/16:
   The graph library (graphlib) is almost finished. However, there is some additional work before release.
   All unit tests are positive only. It needs negative unit testing.

   The GNN library (gnnlib) is still in early stage of development. The GNN can be instantiated but it cannot be
   trained or used for learning tasks.




