def matching_cr(p):
    """
    Input:
        p: a list of pairs, representing the matching
    Output:
        the crossing number of this matching
    Example:
        sage: matching_cr([(1,2),(2,1),(3,3)]);
        1
    """
    counter=0
    for com in Combinations(p,2):
        e1,e2=com
        a,b=e1
        c,d=e2
        if (c-a)*(d-b)<0:
            counter+=1
    return counter

def write_down(B):
    """
    Input:
        B: binary tree in a nested list structure;
    Output:
        write down the numbers in order;
    """
    w=[];
    for a in B:
        if isinstance(a,list):
            w+=write_down(a);
        else:
            w.append(a);
    return w;

def allowable_permutations(B):
    """
    Input:
        B: binary tree in a nested list structure;
    Output:
        a generator of all allowable permutations;
    """
    if isinstance(B,list)==False:
        return [[B]];
    a=B[0];
    b=B[1];
    all_per=[];
    for per_a in allowable_permutations(a):
        for per_b in allowable_permutations(b):
            all_per.append(per_a+per_b);
            all_per.append(per_b+per_a);
    return all_per;

def per_inverse(p):
    n=len(p);
    inv = [0 for i in range(n)]
    for i,pi in enumerate(p):
        inv[pi]=i
    #inv=[i for i in range(n)];
    #inv.sort(key=lambda k:p[k]);
    return inv;

def check_matching(m):
    n=len(m);
    llist=[i for i in range(n)];
    rlist=[i for i in range(n)];
    for pair in m:
        llist.remove(pair[0]);
        rlist.remove(pair[1]);
    return True;

def tangle_cr(b1,b2,m,show_per=False,l_allow=None,r_allow=None):
    """
    Input:
        b1, b2: left and right binary trees;
        m: matching;
        show_per:  True or False;
        l_allow, r_allow: optional input to reduce computation
    Output:
        return the tangle crossing number of (b1,b2,m);
        if show_per==True, also return the permutations;
        if l_allow, r_allow are given, then they will be used as the allowable permutations (instead of computing them);
    """
    n=len(m);
    min_cr=matching_cr(m);
    min_lper=[i for i in range(n)]; # identity
    min_rper=[i for i in range(n)]; # identity
    if l_allow==None:
        l_allow=allowable_permutations(b1);
    if r_allow==None:
        r_allow=allowable_permutations(b2);
    for lper in l_allow:
        for rper in r_allow:
            lper_inv=per_inverse(lper);
            rper_inv=per_inverse(rper);
            new_m=[];
            for pair in m:
                new_m.append((lper_inv[pair[0]],rper_inv[pair[1]]));
            new_cr=matching_cr(new_m);
            if new_cr<min_cr:
                min_cr=new_cr;
                min_lper=lper;
                min_rper=rper;
    if show_per:
        return [min_cr,min_lper,min_rper];
    else:
        return min_cr;

def rank_tree(t,r):
    rank={u:None for u in t}
    rank[r]=0
    stack = list(t.neighbors(r))

    while len(stack)>0:
        v = stack.pop(0)
        for u in t.neighbors(v):
            if rank[u] is None:
                stack.append(u)
            else:
                rank[v]=rank[u]+1
    return rank

def ranked_binary_tree_layout(t,r,perm,rank=None):
    pos = {u:None for u in t}
    if rank is None:
        rank = rank_tree(t,r)
    height = per_inverse(perm)
    #for u in perm:
    #    pos[u] = (0,height[u])

    for v in reversed(list(t.depth_first_search(r))):
        down_nbrs = [ u for u in t.neighbors(v) if rank[v]<rank[u]]
        down_deg = len(down_nbrs)
        if down_deg == 0:
            pos[v] = (0,2*height[v])
        else:
            l,r = down_nbrs
            low = l if pos[l][1]<pos[r][1] else r
            high = r if low==l else l
            x0,y0 = pos[low]
            x1,y1 = pos[high]
            pos[v] = ((x0+x1+y0-y1)/2,(x0-x1+y0+y1)/2)
    return pos


def binary_tree_layout(t,r,rank=None):
    h = 0
    pos = {}
    if rank is None:
        rank = rank_tree(t,r)
    for v in reversed(list(t.depth_first_search(r))):
        down_nbrs = [ u for u in t.neighbors(v) if rank[v]<rank[u]]
        down_deg = len(down_nbrs)
        if down_deg == 0:
            pos[v] = (0,h)
            h=h+2
        else:
            l,r = down_nbrs
            low = l if pos[l][1]<pos[r][1] else r
            high = r if low==l else l
            x0,y0 = pos[low]
            x1,y1 = pos[high]
            pos[v] = ((x0+x1+y0-y1)/2,(x0-x1+y0+y1)/2)
    t.set_pos(pos)

def tree_to_bracket_repr(t,r,rank=None):
    if rank is None:
        rank = rank_tree(t,r)
    childs = [u for u in t.neighbors(r) if rank[u]>rank[r]]
    if len(childs)==0:
        return r
    return [tree_to_bracket_repr(t,childs[0],rank),tree_to_bracket_repr(t,childs[1],rank)]


class Tanglegram:

    def __init__(self,lr,lt,rr,rt,m):
        self.left_root = lr
        self.right_root = rr
        self.left_tree = lt
        self.right_tree = rt

        binary_tree_layout(lt,lr)
        binary_tree_layout(rt,rr)

        self.matching = m
        self.size = len(m)

        self.under_graph = None
        self.g6_string = None
        self.crt=None
        self.default_layout = None
        self.optimal_layout=None

    def underlying_graph(self):

        if self.under_graph is None:
            self.construct_underlying_graph()
        return self.under_graph

    def construct_underlying_graph(self):
        TG = self.left_tree.disjoint_union(self.right_tree)
        TG.add_edges([((0,l),(1,r)) for l,r in self.matching])
        TGpos = {}
        L_pos = self.left_tree.get_pos()
        R_pos = self.right_tree.get_pos()

        reflect = lambda p: (self.size/2.0-p[0],p[1])

        TGpos.update({(0,v):L_pos[v] for v in self.left_tree})
        TGpos.update({(1,v):reflect(R_pos[v]) for v in self.right_tree})

        self.default_layout = TGpos
        self.under_graph = TG

    def graph6_string(self):
        if self.g6_string is None:
            self.compute_graph6_string()
        return self.g6_string

    def compute_graph6_string(self):
        tg = self.underlying_graph()
        lr = (0,self.left_root)
        rr = (1,self.right_root)

        h = tg.copy()
        h.add_edges([(rr,-1)])
        h.add_edges([(lr,-2),(lr,-3)])

        self.g6_string = h.canonical_label().graph6_string()

    def is_isomorphic(self,other):
        return self.graph6_string()==other.graph6_string()

    def plot(self,optimal_layout=False,**kwargs):
        if optimal_layout:
            if self.optimal_layout is None:
                self.compute_optimal_layout()
            return self.underlying_graph().plot(pos=self.optimal_layout,**kwargs)
        return self.underlying_graph().plot(pos=self.default_layout,**kwargs)

    def show(self,optimal_layout=False,**kwargs):
        self.plot(optimal_layout,**kwargs).show()

    def to_bracket_format(self):
        lt = tree_to_bracket_repr(self.left_tree,self.left_root)
        rt = tree_to_bracket_repr(self.right_tree,self.right_root)
        m = copy(self.matching)
        return lt,rt,m

    def crossing_number(self):
        if self.crt is None:
            self.compute_crossing_number()
        return self.crt

    def compute_crossing_number(self):
        lt,rt,m = self.to_bracket_format()
        self.crt = tangle_cr(lt,rt,m)

    def get_optimal_layout(self):
        if self.optimal_layout is None:
            self.compute_optimal_layout()
        return self.optimal_layout

    def compute_optimal_layout(self):
        lt,rt,m = self.to_bracket_format()

        self.crt,left_per,right_per = tangle_cr(lt,rt,m,show_per=True)

        L_pos = ranked_binary_tree_layout(self.left_tree,self.left_root,left_per)
        R_pos = ranked_binary_tree_layout(self.right_tree,self.right_root,right_per)

        TGpos = {}

        reflect = lambda p: (2-p[0],p[1])

        TGpos.update({(0,v):L_pos[v] for v in self.left_tree})
        TGpos.update({(1,v):reflect(R_pos[v]) for v in self.right_tree})
        self.optimal_layout = TGpos

class LaTeXGraph:

    def __init__(self,G):
        self.g = G
        self.latex_string = None
        self.style = None

        with open('latex_skeleton/preamble.tex') as preamblefile:
            self.preamble = preamblefile.read()
        self.postamble = "\end{document}\n"
        self.tikzpicture = "\\begin{{tikzpicture}}\n\n{0}\\end{{tikzpicture}}\n"
        self.tikzpicture_content = None
        self.node_names = { v:'V{0}'.format(v) for v in self.g}

    def latex(self,standalone = True, onlytikzpicture=False, onlytikz=False):

        tikz = self.get_tikzpicture_content()

        if onlytikz:
            return tikz
        if onlytikzpicture:
            return self.tikzpicture.format(tikz)
        if standalone:
            latex_doc = self.preamble
            latex_doc += self.tikzpicture.format(tikz)
            latex_doc += self.postamble
            return latex_doc
        return ""

    def save_to_file(self,path,filename,extension):
        with open(path+filename+extension,'w') as outfile:
            outfile.write(self.latex())

    def set_nodes_latex(self):
        self.node_layer = "\\begin{pgfonlayer}{nodelayer}\n"

        for v in self.g:
            self.node_layer += self.node_latex(v)

        self.node_layer += "\\end{pgfonlayer}\n\n"

    def node_latex(self,v, label=None, style=None):
        p = self.g.get_pos()[v]

        if style is None:
            style='white_v'
        if label is None:
            label = ''
        return "   \\node[style={0}] ({1}) at {2} {{{3}}};\n".format(style,self.node_names[v],p,label)

    def set_edges_latex(self):
        self.edge_layer = "\\begin{pgfonlayer}{edgelayer}\n"
        for u,v in self.g.edges(labels=False):
            self.edge_layer+= self.edge_latex(u,v)
        self.edge_layer+= "\\end{pgfonlayer}\n\n"

    def edge_latex(self,u,v,style=None):
        u_name = self.node_names[u]
        v_name = self.node_names[v]
        if style is None:
            style = 'simple'
        return "   \\draw[style={0}] ({1}) to ({2});\n".format(style,u_name, v_name)

    def set_tikzpicture_content(self):
        self.set_nodes_latex()
        self.set_edges_latex()
        self.tikzpicture_content = self.node_layer+self.edge_layer

    def get_tikzpicture_content(self):
        if self.tikzpicture_content is None:
            self.set_tikzpicture_content()
        return self.tikzpicture_content

def flatten(l):
    if isinstance(l[0],list):
        for e in flatten(l[0]):
            yield e
    else:
        yield l[0]
    if isinstance(l[1],list):
        for e in flatten(l[1]):
            yield e
    else:
        yield l[1]
node_name = lambda l:''.join([str(x) for x in sorted(flatten(l))])

def get_tree(b):
    root_node = node_name(b)
    if isinstance(b[0],list):
        lR, lT =  get_tree(b[0])
    else:
        lR = b[0]
        lT = Graph({lR:[]})
        lT.set_pos({lR:(0,2*lR)})
    if isinstance(b[1],list):
        rR, rT =  get_tree(b[1])
    else:
        rR = b[1]
        rT = Graph({rR:[]})
        rT.set_pos({rR:(0,2*rR)})
    T = lT.union(rT)
    T.add_edges([[root_node,lR],[root_node,rR]])
    max_h = max([max([y for x,y in rT.get_pos().values()]),max([y for x,y in lT.get_pos().values()])])
    min_h = min([min([y for x,y in rT.get_pos().values()]),min([y for x,y in lT.get_pos().values()])])
    root_pos = (-(len(root_node)-1),(max_h+min_h)/2)
    Tpos = {}
    Tpos.update(lT.get_pos())
    Tpos.update(rT.get_pos())
    Tpos[root_node]=root_pos
    T.set_pos(Tpos)
    return root_node,T

def get_tanglegram(b1,b2,m):
    lr,LT = get_tree(b1)
    rr,RT = get_tree(b2)
    TG = LT.disjoint_union(RT)
    TG.add_edges([((0,l),(1,r)) for l,r in m])
    TGpos = {}
    Lpos = LT.get_pos()
    Rpos = RT.get_pos()

    reflect = lambda p: (2-p[0],p[1])

    TGpos.update({(0,v):Lpos[v] for v in LT})
    TGpos.update({(1,v):reflect(Rpos[v]) for v in RT})

    TG.set_pos(TGpos)
    return (lr,rr,TG)

def get_tanglegram_trees(lr,LT,rr,RT,m):
    TG = LT.disjoint_union(RT)
    TG.add_edges([((0,l),(1,r)) for l,r in m])
    return ((0,lr),(1,rr),TG)

def tanglegram_g6(T):
    rr,lr,tg = T

    h = tg.copy()
    h.add_edges([(rr,-1)])
    h.add_edges([(lr,-2),(lr,-3)])

    return h.canonical_label().graph6_string()

def tanglegram_isomorphism(T1,T2):
    rr1,lr1,tg1 = T1
    rr2,lr2,tg2 = T2

    h = tg1.copy()
    h.add_edges([(rr1,-1)])
    h.add_edges([(lr1,-2),(lr1,-3)])

    g = tg2.copy()
    g.add_edges([(rr2,-1)])
    g.add_edges([(lr2,-2),(lr2,-3)])

    return h.is_isomorphic(g)