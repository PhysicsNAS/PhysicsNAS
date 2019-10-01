import graphviz
from graphviz import Digraph

def plot_genotype(genotype, file_name=None, figure_dir='./network_structure', save_figure=False):
    
    # Set graph style
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='rounded,filled', shape='box', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='1', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])
    
    input_node_names = ['x', 'x_dup', 'y_phy', '[x, y_phy]']
    output_name = 'y_hat'

    # All input nodes
    for input_node_name in input_node_names:
        g.node(input_node_name, fillcolor='mediumaquamarine')
    
    # Number of inner nodes
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    # All inner nodes
    for i in range(steps - 1):
        g.node(str(i), fillcolor='darkgoldenrod1')
    
    # Output node
    g.node(output_name, fillcolor='cornflowerblue')

    # Add edges 
    # Edge direction: u ---> v
    # Genotype: operation, u
    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j < len(input_node_names):
                u = input_node_names[j]
            else:
                u = str(j - len(input_node_names))
                
            if i == steps - 1:
                v = output_name
            else:
                v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")
    
    # Save the figure
    if save_figure:
        g.render(file_name, view=False, directory=figure_dir)
        
    return g