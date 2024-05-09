Implementation of EoH (Liu et al., 2024). Code adapted from [official source code](https://github.com/FeiLiu36/EoH/tree/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/):

<pre>
/baselines/eoh
|-- __init__.py
|-- eoh_adapter.py
|-- problem_adapter.py
|-- readme.md
`-- original
    |-- eoh.py               <- <a href="https://github.com/FeiLiu36/EoH/tree/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/methods/eoh/eoh.py">/eoh/methods/eoh/eoh.py</a>
    |-- eoh_evolution.py     <- <a href="https://github.com/FeiLiu36/EoH/tree/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/methods/eoh/eoh_evolution.py">/eoh/methods/eoh/eoh_evolution.py</a>
    |-- eoh_interface_EC.py  <- <a href="https://github.com/FeiLiu36/EoH/tree/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/methods/eoh/eoh_interface_EC.py">/eoh/methods/eoh/eoh_interface_EC.py</a>
    |-- pop_greedy.py        <- <a href="https://github.com/FeiLiu36/EoH/blob/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/methods/management/pop_greedy.py">/eoh/methods/management/pop_greedy.py</a>
    |-- prob_rank.py         <- <a href="https://github.com/FeiLiu36/EoH/blob/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/methods/selection/prob_rank.py">/eoh/methods/selection/prob_rank.py</a>
    |-- interface_LLM.py     <- <a href="https://github.com/FeiLiu36/EoH/blob/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/llm/api_general.py">/eoh/llm/api_general.py</a>
    |-- getParas.py          <- <a href="https://github.com/FeiLiu36/EoH/blob/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/utils/getParas.py">/eoh/utils/getParas.py</a>
    `-- prompts
        |-- bpp_online.py    <- <a href="https://github.com/FeiLiu36/EoH/blob/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/problems/optimization/bp_online/prompts.py">/eoh/problems/optimization/bp_online/prompts.py</a>
        `-- tsp_greedy.py    <- <a href="https://github.com/FeiLiu36/EoH/blob/d7c5a69f4f6b70b475c5ef942f9c72675fe9ac71/eoh/src/eoh/problems/optimization/tsp_greedy/prompts.py">/eoh/problems/optimization/tsp_greedy/prompts.py</a>
</pre>

References:
- Liu, F., Tong, X., Yuan, M., Lin, X., Luo, F., Wang, Z., Lu, Z., & Zhang, Q. (2024). Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Mode (arXiv:2401.02051). arXiv. https://doi.org/10.48550/arXiv.2401.02051
