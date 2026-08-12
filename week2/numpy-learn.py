import numpy as np

# function to check the version of numpy
def versionCheck():
    print("Numpy version:", np.__version__)

# check dimensions of various arrays
def checkDimensions(arr: np.ndarray):
    print("Array:\n", arr)
    print("Number of dimensions:", arr.ndim)

# function to create various numpy arrays
def createArrays():
    # creating an array from a tuple
    arrFromTuple = np.array((1, 2, 3, 4, 5))
    print("Array from tuple:", arrFromTuple)
    checkDimensions(arrFromTuple)

    # creating a 0D array (scalar)
    arr0D = np.array(42)
    print("0D Array (scalar):", arr0D)
    checkDimensions(arr0D)

    # creating a 1D array
    arr1D = np.array([1, 2, 3, 4, 5])
    print("1D Array:", arr1D)
    checkDimensions(arr1D)

    # creating a 2D array
    arr2D = np.array([[1, 2, 3], [4, 5, 6]])
    print("2D Array:\n", arr2D)
    checkDimensions(arr2D)

    # creating a 3D array
    arr3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print("3D Array:\n", arr3D)
    checkDimensions(arr3D)

versionCheck()
createArrays()