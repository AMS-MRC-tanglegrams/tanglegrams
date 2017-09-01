# Codes for generating tanglegrams of size n uniformly

This resipository stores the codes for generating random tanglegrams uniformly (up to isomorphism).  The algotirhm is based on [1].

[1] Sara C. Billey, Matjaž Konvalinka, Frederick A. Matsen IV.  On the enumeration of tanglegrams and tangled chains.  Journal of Combinatorial Theory, Series A 146 (2017) 239–263.

## How to use

If you have SageMath installed on your machine and have internet access, you may use

    URL="https://raw.githubusercontent.com/jephianlin/tanglegrams/master/src/"
    files=["tanglegram_class.sage","random_tanglegram.sage","simulation.sage"]
    for f in files:
        load(URL+f);

Otherwise, you may run this on SageMathCell by clicking [here](https://sagecell.sagemath.org/?z=eJxFjL0KgzAURnfBd5BMSksydbH4Bk5Sp1LkNsYkJT-Se8XXryit33jOx-m7tmGGaMZaiAQr15bM8l5QJRkDqUBcRi8-ajYWgrNBEATtlE7gUXhAUklgkoLl2WSdwubJzscgHSByBK3YlSUIY_TDqX8CrV8ckI3hIK-tFVMxFTYUe7TOs2KbizCWfddepup-kDzr9ujj3yxvFUcT17L6At_zSsk=&lang=sage)
