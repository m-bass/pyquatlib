 # pyquatlib
 
 Just another library for quaternions.
 With focus on usability.
 
 ## Installation
 
 Assuming you use `pip` to manage your python installation,
 you can install this package simply as
 
 ```sh
 pip install pyquatlib
 ```
 
 In order to install a virtual environment,
 change to the code directory, and run
 
 ```sh
 make init
 ```
 
 ## Basic usage
 
 The full documentation can be generated with
 
 ```sh
 make docs
 ```
 
  The following are mostly for the purposes of example.
 
 There are several ways to create a Quaternion:
 ```python
 >>> from quaternions import Quaternion
 >>> import numpy as np
 
 >>> Quaternion(np.array([1, -2, 3]))
 Quaternion: ((0, array([ 1, -2,  3])))
 >>> Quaternion((0, np.array([1, 2, 3])))
 Quaternion: ((0, array([1, 2, 3])))
 ```
 
 Left and right operations with compatible data structures are supported.
 ```python
 >>> data = np.full((4,3), [1,2,3])
 >>> data
 array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
 
 >>> data - Quaternion([2, 1, 1, 1])
 [Quaternion: ((-2, array([0, 1, 2]))),
  Quaternion: ((-2, array([0, 1, 2]))),
  Quaternion: ((-2, array([0, 1, 2]))),
  Quaternion: ((-2, array([0, 1, 2])))]
 
 >>>  Quaternion([1, 0 , -1, 0]) + data
 [Quaternion: ((1, array([1, 1, 3]))),
  Quaternion: ((1, array([1, 1, 3]))),
  Quaternion: ((1, array([1, 1, 3]))),
  Quaternion: ((1, array([1, 1, 3])))]
 
 >>> data * Quaternion(np.array([1, 2, 3, 4]))
 [Quaternion: ((-20, array([0, 4, 2]))),
  Quaternion: ((-20, array([0, 4, 2]))),
  Quaternion: ((-20, array([0, 4, 2]))),
  Quaternion: ((-20, array([0, 4, 2])))]

 >>> Quaternion([0, 1 , -1, -1]) / data
 [Quaternion: ((-0.2857142857142857, array([ 0.07142857,  0.28571429, -0.21428571]))),
  Quaternion: ((-0.2857142857142857, array([ 0.07142857,  0.28571429, -0.21428571]))),
  Quaternion: ((-0.2857142857142857, array([ 0.07142857,  0.28571429, -0.21428571]))),
  Quaternion: ((-0.2857142857142857, array([ 0.07142857,  0.28571429, -0.21428571])))]
 ```
 
 
 Several conversion functions from compatible data structures are also included.
 For example, to convert an (n,3) array to a list of Quaternions, use `to_quaternion`:
 
 ```python
 >>> vectors = np.random.rand(100, 3)
 >>> vectors
 array([[0.02219696, 0.61847575, 0.68714365],
        [0.56104393, 0.3529833 , 0.8188565 ],
        ...
        [0.38190174, 0.59151826, 0.37396559]])
 
 >>> Quaternion.to_quaternion(vectors)
 [Quaternion: ((0.0, array([0.02219696, 0.61847575, 0.68714365]))),
  Quaternion: ((0.0, array([0.56104393, 0.3529833 , 0.8188565 ]))),
  Quaternion: ((0.0, array([0.84834741, 0.02304604, 0.33619428]))),
  ...
  Quaternion: ((0.0, array([0.38190174, 0.59151826, 0.37396559])))]
 ```
 
 ## Bug reports and feature requests
 
 Bug reports and feature requests are entirely welcome.
 The best way to do this is to open an [issue on this code's github
 page](https://github.com/m-bass/pyquatlib/issues).  For bug reports,
 please try to include a minimal working example demonstrating the
 problem.
 
 [Pull requests](https://help.github.com/articles/using-pull-requests/)
 are also entirely welcome, of course, if you have an idea where the
 code is going wrong, or have an idea for a new feature that you know
 how to implement.
 
 This code is routinely tested on recent versions of Python 3.\*
 Test coverage is quite complete.
