import copy
from graphviz import Digraph


def model_visualize(model, save_dir):
    gene_cell = copy.deepcopy(model).module.cpu().genotype()
    visualize(gene_cell, save_dir)


def visualize(gene_cell, save_dir):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="Helvetica"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.6', width='0.6',
                       penwidth='2', fontname="times"),
        engine='dot'
    )
    g.body.extend(['rankdir=LR'])  # Adjust spacing between nodes
    pre_cell_color = '#b2e0b2'  # Soft light green
    cur_cell_color = '#ffffb2'  # Soft light yellow
    operation_edge_color = 'black'  # Change to black
    connection_edge_color = 'black'  # Keep the connection edges gray
    for cell_id in range(len(gene_cell)):
        gene_cell_i = gene_cell[cell_id]
        pre_pre_cell = "Pre_pre_cell"
        pre_cell = "Pre_cell"
        if cell_id == 0:
            pre_pre_cell = 'stem0'
            pre_cell = 'stem1'
        elif cell_id == 1:
            pre_pre_cell = 'stem1'
            pre_cell = 'cell_0'
        elif cell_id > 1:
            pre_pre_cell = 'cell_{}'.format(cell_id - 2)
            pre_cell = 'cell_{}'.format(cell_id - 1)
        cur_cell = 'cell_{}'.format(cell_id)
        # Add pre and current cells
        g.node(pre_pre_cell, fillcolor=pre_cell_color)
        g.node(pre_cell, fillcolor=pre_cell_color)
        node_num = len(gene_cell_i) // 2
        for i in range(node_num):
            g.node(name='C{}_N{}'.format(cell_id, i), fillcolor='lightblue', fontsize='16')
        for i in range(node_num):
            for k in [2 * i, 2 * i + 1]:
                op, j = gene_cell_i[k]
                if op != 'none':
                    if j == 1:
                        u = pre_pre_cell
                        v = 'C{}_N{}'.format(cell_id, i)
                        g.edge(u, v, label=op, color=operation_edge_color, style='solid')
                    elif j == 0:
                        u = pre_cell
                        v = 'C{}_N{}'.format(cell_id, i)
                        g.edge(u, v, label=op, color=operation_edge_color, style='solid')
                    else:
                        u = 'C{}_N{}'.format(cell_id, j - 2)
                        v = 'C{}_N{}'.format(cell_id, i)
                        g.edge(u, v, label=op, color=connection_edge_color, style='solid')
        g.node(cur_cell, fillcolor=cur_cell_color)
        for i in range(node_num):
            g.edge('C{}_N{}'.format(cell_id, i), cur_cell, color=connection_edge_color)
    g.render(save_dir, view=False)
