import random
from collections import defaultdict as dd

# tanglegram_class.sage essentially contains the code used in the catalogue to generate
# all tanglgrams. Beware, heavily undocumented code!
# In particular it contains the Tanglegram class and also the routine to lay out trees.
load('tanglegram_class.sage')

def binary_partitions(n):
    """
    binary_partitions is part of PADS, which is licensed under the MIT Licence (http://opensource.org/licenses/MIT)
    Copyright (c) 2002-2015, David Eppstein

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

    Source: https://www.ics.uci.edu/~eppstein/PADS/
    File:   IntegerPartitions.py
    Generate partitions of n into powers of two, in revlex order.
    Knuth exercise 7.2.1.4.64.
    The average time per output is constant.
    But this doesn't really solve the exercise, because it isn't loopless...
    """

    # Generate the binary representation of n
    if n < 0:
        return
    pow = 1
    sum = 0
    while pow <= n:
        pow <<= 1
    partition = []
    while pow:
        if sum+pow <= n:
            partition.append(pow)
            sum += pow
        pow >>= 1

    # Find all partitions of numbers up to n into powers of two > 1,
    # in revlex order, by repeatedly splitting the smallest nonunit power,
    # and replacing the following sequence of 1's by the first revlex
    # partition with maximum power less than the result of the split.

    # Time analysis:
    #
    # Each outer iteration increases len(partition) by at most one
    # (only if the power being split is a 2) and each inner iteration
    # in which some ones are replaced by x decreases len(partition),
    # so the number of those inner iterations is less than one per
    # output.
    #
    # Each time a power 2^k is split, it creates two or more 2^{k-1}'s,
    # all of which must eventually be split as well.  So, it S_k denotes
    # the number of times a 2^k is split, and X denotes the total
    # number of outputs generated, then S_k <= X/2^{k-1}.
    # On an outer iteration in which 2^k is split, there will be k
    # inner iterations in which x is halved, so the total number
    # of such inner iterations is <= sum_k k*X/2^{k-1} = O(X).
    #
    # Therefore the overall average time per output is constant.

    last_nonunit = len(partition) - 1 - (n&1)
    while True:
        yield partition
        if last_nonunit < 0:
            return
        if partition[last_nonunit] == 2:
            partition[last_nonunit] = 1
            partition.append(1)
            last_nonunit -= 1
            continue
        partition.append(1)
        x = partition[last_nonunit] = partition[last_nonunit+1] = \
            partition[last_nonunit] >> 1    # make the split!
        last_nonunit += 1
        while x > 1:
            if len(partition) - last_nonunit - 1 >= x:
                del partition[-x+1:]
                last_nonunit += 1
                partition[last_nonunit] = x
            else:
                x >>= 1

def write_down(B):
    """
    Input:
        B: binary tree in a nested list structure;
    Output:
        write down the numbers in order;
    """
    if isinstance(B,list):
        w=[];
        for a in B:
            if isinstance(a,list):
                w+=write_down(a);
            else:
                w.append(a);
        return w;
    else:
        return [B];

def shift(T,k):
    """
    Input:
        T: A binary tree rpresented as nested lists of length 2.
           Leaves are assumed to be numbers.
        k: a number to shift the leaves by.
    Output:
        None; warning, this function edits the input T.
    """
    if isinstance(T,list):
        return [shift(T[0],k),shift(T[1],k)];
    else:
        return T+k;

def BinaryTreeSize(T):
    """
    Input:
        T: binary tree;
    Output:
        the number of leaves of T;
        if T is not a list, returen 1;
    """
    if isinstance(T,list):
        return len(write_down(T));
    else:
        return 1;

def BinaryTree_geq(T,S):
    """
    Input:
        T, S: binary trees;
    Output:
        return T geq S or not;
    """
    n1=BinaryTreeSize(T);
    n2=BinaryTreeSize(S);
    if n1>n2:
        return True;
    elif n1<n2:
        return False;
    elif n1==1 and n2==1:
        return True;
    else:
        if BinaryTree_geq(T[0],T[1]):
            T0=T[0];
            T1=T[1];
        else:
            T0=T[1];
            T1=T[0];
        if BinaryTree_geq(S[0],S[1]):
            S0=S[0];
            S1=S[1];
        else:
            S0=S[1];
            S1=S[0];
        if BinaryTree_geq(S0,T0)==False: #equivalently T0>S0
            return True;
        elif BinaryTree_geq(T0,S0)==False: #equivalently T0<S0
            return False;
        else:
            return BinaryTree_geq(T1,S1);

def BinaryTreeIsomorphism(T,S):
    """
    Input:
        T, S: two binary trees;
    Output:
        return the isomorphism from T to S, otherwise return False;
    """
    n1=BinaryTreeSize(T);
    n2=BinaryTreeSize(S);
    if n1==1 and n2==1:
        return [(T,S)];
    if n1!=n2:
        return False;
    Iso00=BinaryTreeIsomorphism(T[0],S[0]);
    Iso01=BinaryTreeIsomorphism(T[0],S[1]);
    Iso10=BinaryTreeIsomorphism(T[1],S[0]);
    Iso11=BinaryTreeIsomorphism(T[1],S[1]);
    if Iso00 and Iso11:
        return Iso00+Iso11;
    if Iso01 and Iso10:
        return Iso01+Iso10;
    return False;

def line_representation(per):
    new_per=sorted(per,key=lambda k:k[0]);
    line_rep="";
    for m in new_per:
        line_rep+="%s"%m[1];
    return line_rep;

def cycle_representation(per):
    """
    Input:
        a permutation of the matching structure;
        e.g., [(0,1),(1,2),(2,0)];
    Output:
        the cycle representation of per
        under the dictionary structure: {k: [cycles of length k]};
        each cycle is stored as a list;
    """
    dom=sorted(set([x for x,y in per]));
    #I assume the given permutation does not ignore any maps of the form (i,i);
    n=len(dom);
    all_cyc={k:[] for k in range(1,n+1)};
    while dom:
        cyc=[];
        a=dom[0];
        b=a;
        cyc.append(b);
        again=True;
        while again:
            for x,y in per:
                if b==x and y!=a:
                    b=y;
                    cyc.append(b);
                    break;
                if b==x and y==a:
                    again=False;
        all_cyc[len(cyc)].append(cyc);
        for v in cyc:
            dom.remove(v);
    return all_cyc;

def per_prod(p1,p2):
    """
    Input:
        p1,p2: two permutations;
    Output:
        the product p1*p2;
    """
    q1=copy(p1);
    q2=copy(p2);
    p_prod=[];
    domain = sorted(set([x for x,y in q1]+[x for x,y in q2]));
    for x in domain:
        y = x
        for X,Y in p2:
            if y==X:
                y=Y
                break
        for X,Y in p1:
            if y==X:
                y = Y
                break
        p_prod.append((x,y))
    return p_prod;

def RandomAutomorphism_of_BinaryTree(T): #Algorithm 1
    """
    Input:
        T: a binary tree in the nested list structure;
    Output:
        a random automorphism of T chosen uniformly
    """
    if BinaryTreeSize(T)==1:
        return [(T,T)];
    else:
        T0,T1=T;
        Iso=BinaryTreeIsomorphism(T0,T1);
        per0=RandomAutomorphism_of_BinaryTree(T0);
        per1=RandomAutomorphism_of_BinaryTree(T1);
        if Iso:
            flip=[(m[0],m[1]) for m in Iso]+[(m[1],m[0]) for m in Iso];
            flip_or_not=random.choice([0,1]);
            if flip_or_not==0:
                return per0+per1;
            if flip_or_not==1:
                return per_prod(flip,per0+per1);
        else:
            return per0+per1;

from collections import defaultdict
from bisect import bisect_left
def multiplicities( partition ):
    """
    Input:
        partition: a list of numbers representing a partition of n
    Output:
        mutiplicities: a dictionary k -> v where k is a part in the
        input partition and v is the multiplicity of k in the partition
    """
    multiplicities = defaultdict(lambda:0)
    for x in partition:
        multiplicities[x]+=1
    return multiplicities

def zed(partition):
    """
    Input:
        partition: a list of numbers representing a partition of n
    Output:
        z: see equation (1), page 5 in tanglegram_enumeration.pdf
    """
    mult = multiplicities( partition )
    z=1
    for b in mult:
        m = mult[b]
        z *= b**m*factorial(m)
    return z

def cue(partition, z=None):
    """
    Input:
        partition: a list of numbers representing a partition of n
        (optional) z: see (1) page 5 in tanglegram_enumeration.pdf
    Output:
        q: See displayed equation, page 14 at tanglegram_enumeration.pdf
    """
    if z is None:
        z = zed(partition)

    k = len(partition)
    product = 1
    for i in range(2,k+1):
        product *= (2*(sum(partition[i-1:]))-1)
    return product/z

def weighted_random_binary_partition(n):
    """
    Input:
        n: returns a binary partition of n with probability proportional to z*q^2
        For definition of z: see equation (1), page 5 in tanglegram_enumeration.pdf
        For definition of q: see displayed equation, page 14 in tanglegram_enumeration.pdf
        The sum over all binary partitions of n of z*q^2 equals the number of tanglegrams on
        n points.
    Output:
        partition: a list of numbers representing a binary partition of n
    """
    zed_cue_square = []
    partitions = []
    total = 0
    for p in binary_partitions(n):
        z = zed(p)
        q = cue(p,z)
        zqs = z*(q**2)
        zed_cue_square.append(zqs)
        partitions.append(copy(p))
        total += zqs
    X = GeneralDiscreteDistribution([zqs/total for zqs in zed_cue_square])
    partition = partitions[X.get_random_element()]
    return partition

def generate_random_partition_subdivision(l):
    """
    Input:
        l: a list representing a binary partition of n
    Output:
        subdivision: a tuple containing two lists that define a subdivision of the input partition.

    """
    tcuel = 2*cue(l);
    probabilities=[];
    subdivisions = [];
    for j in range(1,len(l)):
        for com in Combinations(l,j):
            com_bar=copy(l);
            for i in com:
                com_bar.remove(i);
            probabilities.append(cue(com)*cue(com_bar)/tcuel);
            subdivisions.append((com,com_bar));

    if all([x%2==0 for x in l]):
        com = [x/2 for x in l];
        probabilities.append(cue(com)/tcuel);
        subdivisions.append((com,com));

    X = GeneralDiscreteDistribution(probabilities);
    subdivision = subdivisions[X.get_random_element()];
    return subdivision;

def RandomBinaryTree_with_Aut(l):  #Algorithm 2
    """
    Input:
        l: a binary partition;
    Output:
        return [T,w]
        where T is a binary tree and
        w is its automorphism with given binary partition;
    """

    if sum(l)==1:
        return (0,[(0,0)]); #notice that 0 is the binary tree of one vertex

    lam1,lam2 = generate_random_partition_subdivision(l);
    lam1.sort();
    lam2.sort();
    lam_half = [x/2 for x in l];
    if lam1==lam2 and lam1==lam_half:
        T1,w2=RandomBinaryTree_with_Aut(lam1);
        w1=RandomAutomorphism_of_BinaryTree(T1);
        n1=BinaryTreeSize(T1);
        flip=[(i,i+n1) for i in range(n1)]+[(i+n1,i) for i in range(n1)]; # assuming T1 is labeled as 0 ~ n-1
        T2=shift(T1,n1);
        w2=[(x+n1,y+n1) for x,y in w2]
        T=[T1,T2];
        w=flip;
        w1_inv=[(y,x) for x,y in w1];
        for per in [w1,flip,w1_inv,flip,w2]:
                    w=per_prod(w,per);
        return (T,w);
    else:
        T1,w1=RandomBinaryTree_with_Aut(lam1);
        T2,w2=RandomBinaryTree_with_Aut(lam2);
        if BinaryTree_geq(T1,T2)==False:
            T1,T2 = T2,T1;
            w1,w2 = w2,w1;
        n1=BinaryTreeSize(T1);
        T2=shift(T2,n1);
        T=[T1,T2];
        w2=[(x+n1,y+n1) for x,y in w2];
        w=w1+w2;
        return (T,w);


def traverse_nested_list(b,T):
    """
    Intput:
        b: binary tree in bracket notation, nested lists of size 2.
        T: the Graph that will be the tree represented by b, note T is constructed recursively and inplace!
    Output:
        x: the root of the current subtree.
    Warning:
       This function does not handle nicely the case where b is a vertex itself, beware!
    """
    x = T.add_vertex()
    left_root = traverse_nested_list(b[0],T) if isinstance(b[0],list) else b[0]
    right_root = traverse_nested_list(b[1],T) if isinstance(b[1],list) else b[1]
    T.add_edges([(x,left_root),(x,right_root)])
    return x

def bracket_to_graph(b):
    """
    Input:
        b: a binary tree in bracket notation
    Output:
        T: the tree T represented by b as a Graph
        r: the root of T (ie, the degree 2 vertex of T)
    """
    leaves = write_down(b)
    T = Graph()
    T.add_vertices(leaves)
    root = traverse_nested_list(b,T)
    return root,T


def random_base_change(u,v):
    """
    Input:
        u,v: two permutation under matching structure;
    Output:
        return a random permutation w with u=w v w^{-1} uniformly;
    """
    u_cycles=cycle_representation(u);
    v_cycles=cycle_representation(v);
    cycle_matching=[];
    # randomly match a cycle in u to a cycle in v with the same length;
    for k in u_cycles.keys():
        u_kcycles=u_cycles[k];
        v_kcycles=v_cycles[k];
        while u_kcycles:
            u_cyc=u_kcycles.pop();
            v_cyc=random.choice(v_kcycles);
            v_kcycles.remove(v_cyc);
            cycle_matching.append((u_cyc,v_cyc));
    per=[];
    while cycle_matching:
        u_cyc,v_cyc=cycle_matching.pop();
        d=len(u_cyc);
        i=random.choice(range(d)); #how much to shift v_cyc
        v_cyc=v_cyc[i:]+v_cyc[:i];
        for j in range(d):
            per.append((u_cyc[j],v_cyc[j]));
    #now we have v=per u per^{-1};
    #so what I want is per_inv;
    per_inv=[(y,x) for x,y in per];
    return per_inv;

def RandomTanglegram(n):
    """
    Input:
        n: integer;
    Output:
        a uniformly random tanglegram of order n;
    """
    l=weighted_random_binary_partition(n);
    T,u=RandomBinaryTree_with_Aut(l);
    S,v=RandomBinaryTree_with_Aut(l);
    w=random_base_change(u,v);

    lr,L = bracket_to_graph(T);
    rr,R = bracket_to_graph(S);
    Tangle = Tanglegram(lr,L,rr,R,w);
    return Tangle;