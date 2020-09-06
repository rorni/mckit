# mckit
mckit is a tool to manipulate complex MCNP models. When the model is rather complex and its description occupies thousands of text lines it becomes hard to do some modifications.
Especially almost impossible to do integration of the model. Mckit automates integration process. 

For now it is rather framework than easy-to-use tool, and requires user to do some python programming. However, now we know main usecases, 
implementation of user-friendly interface requires some time. 

## Installation

For performance purposes geometry manipulation kernel is implemented in C. So, installation from sources requires compilation. However, there are prebuild 
python wheels for python 3.6 and 3.7 (you can find them in build folder). 

## Tutorial

There is a python notebook with some examples in tutorial folder. Also some very brief description you can find there. All functions and classes have doc strings. 


