# Generating tanglegrams of size n uniformly at random

This repository stores the code for generating random tanglegrams
uniformly (up to isomorphism).  This contains an implementation of
Algorithm 1, 2, and 3 from [1, p. 252-253]

[1] Sara C. Billey, Matjaž Konvalinka, Frederick A. Matsen IV.
    On the enumeration of tanglegrams and tangled chains.
    Journal of Combinatorial Theory, Series A 146 (2017) 239–263.

## How to generate a random tanglegram

If you have SageMath installed on your machine and have internet
access, you may use

    load("https://raw.githubusercontent.com/AMS-MRC-tanglegrams/tanglegrams/master/src/sagecell_random_tanglegram.sage")


Otherwise, you may run this on SageMathCell by clicking
[here](http://sagecell.sagemath.org/?z=eJxNyjEOwjAMQNG9Ui_RKRkabwxsiLlL6V6Z1EqQkhjZrnp9BAOwfen9wri5IZs99QwgeIT0sLzfdyWJ3IyahcgVLtNtnObraNhSoSRYFf67ohoJqERQTBSplFWwbVzX3xbeNPi-67v5Y8uX3MkHzXw4_wKHljTw&lang=sage).
