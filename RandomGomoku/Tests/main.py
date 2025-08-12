import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../..")

path = os.getcwd()
print(path)

sys.path.append(path)

print(os)

from RandomGomoku.Tests.Board.test2 import Test

if __name__ == "__main__":
    Test()
