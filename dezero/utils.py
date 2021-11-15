import os
import subprocess


def _dot_var(v, verbose=False):
    dot_var = '{}[label="{}", color = orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

# 基本はbackwardメソッドと同じ
def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            #　ノードを辿る順番は今回問題ではないので、今回はコメンタアウト
            # funcs.sort(key= lambda x: x.generation)
            seen_set.add(f)
    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        funcs = funcs.pop(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' +  txt + '}'

def plot_dot_graph(output, verbose = True, to_file = 'graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # ①dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser('~'), 'dezero')
    if not os.path.exists(tmp_dir):
        # ~/.dezeroディレクトリがなかったら作成
        os.mkdir(tmp_dir)
    graph_path=os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    #②dotコマンドを呼ぶ
    extension = os.path.splittext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd,shell=True)