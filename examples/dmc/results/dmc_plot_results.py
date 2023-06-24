
import plotify as pl


wandb_runs = {
    'sac': {
        'ball_in_cup-catch': [
            'arnolds/cherry-dmc-benchmarks/kld2vclj',
            'arnolds/cherry-dmc-benchmarks/zcb1amw6',
            'arnolds/cherry-dmc-benchmarks/gh47b0dx',
            'arnolds/cherry-dmc-benchmarks/ezu4bs7k',
            'arnolds/cherry-dmc-benchmarks/m5itjax6',
        ],
        'cartpole-swingup': [
            'arnolds/cherry-dmc-benchmarks/b7nmhtxc',
            'arnolds/cherry-dmc-benchmarks/spcdheu6',
            'arnolds/cherry-dmc-benchmarks/vc54uotu',
            'arnolds/cherry-dmc-benchmarks/a5hxxy16',
            'arnolds/cherry-dmc-benchmarks/fv36e7qq',
        ],
        'reacher-easy': [
            'arnolds/cherry-dmc-benchmarks/84243w7y',
            'arnolds/cherry-dmc-benchmarks/hssqv183',
            'arnolds/cherry-dmc-benchmarks/3swmggei',
            'arnolds/cherry-dmc-benchmarks/8zed6i2r',
            'arnolds/cherry-dmc-benchmarks/9maccjmh',
        ],
        'finger-spin': [
            'arnolds/cherry-dmc-benchmarks/0mszo1vc',
            'arnolds/cherry-dmc-benchmarks/e6hboct9',
            'arnolds/cherry-dmc-benchmarks/u5kqkstb',
            'arnolds/cherry-dmc-benchmarks/y6gvhbz0',
            'arnolds/cherry-dmc-benchmarks/2xybpx7d',
        ],
        'cheetah-run': [
            'arnolds/cherry-dmc-benchmarks/ynb9st3s',
            'arnolds/cherry-dmc-benchmarks/9lsppgc9',
            'arnolds/cherry-dmc-benchmarks/ynpchfo4',
            'arnolds/cherry-dmc-benchmarks/o66mx21f',
            'arnolds/cherry-dmc-benchmarks/6v8fgtf0',
        ],
        'walker-walk': [
            'arnolds/cherry-dmc-benchmarks/qobvuu33',
            'arnolds/cherry-dmc-benchmarks/15yafe7v',
            'arnolds/cherry-dmc-benchmarks/igllgewl',
            'arnolds/cherry-dmc-benchmarks/98trezzi',
            'arnolds/cherry-dmc-benchmarks/t6b8sjbx',
        ],
    },
    'drq': {
        'ball_in_cup-catch': [
            'arnolds/cherry-dmc-benchmarks/4vijenen',
            'arnolds/cherry-dmc-benchmarks/wlrh2lpu',
            'arnolds/cherry-dmc-benchmarks/1rvhzmvw',
            'arnolds/cherry-dmc-benchmarks/9xrq9mw1',
            'arnolds/cherry-dmc-benchmarks/5htva4zn',
        ],
        'cartpole-swingup': [
            'arnolds/cherry-dmc-benchmarks/qm37j6v8',
            'arnolds/cherry-dmc-benchmarks/k17ww9np',
            'arnolds/cherry-dmc-benchmarks/02vb7m61',
            'arnolds/cherry-dmc-benchmarks/2xmoitk7',
            'arnolds/cherry-dmc-benchmarks/y10nl03t',
        ],
        'reacher-easy': [
            'arnolds/cherry-dmc-benchmarks/e6gf1u2q',
            'arnolds/cherry-dmc-benchmarks/9jad58tv',
            'arnolds/cherry-dmc-benchmarks/hww2ptl0',
            'arnolds/cherry-dmc-benchmarks/75m6w2eg',
            'arnolds/cherry-dmc-benchmarks/imoy1oit',
        ],
        'finger-spin': [
            'arnolds/cherry-dmc-benchmarks/guq5pkkm',
            'arnolds/cherry-dmc-benchmarks/0x2g48c4',
            'arnolds/cherry-dmc-benchmarks/jnhqfkq5',
            'arnolds/cherry-dmc-benchmarks/eidsk54j',
            'arnolds/cherry-dmc-benchmarks/g50bxhg7',
        ],
        'cheetah-run': [
            'arnolds/cherry-dmc-benchmarks/3ned5elf',
            'arnolds/cherry-dmc-benchmarks/5gag70bm',
            'arnolds/cherry-dmc-benchmarks/76x2yrbc',
            'arnolds/cherry-dmc-benchmarks/hgvg8fc3',
            'arnolds/cherry-dmc-benchmarks/aotdtqf4',
        ],
        'walker-walk': [
            'arnolds/cherry-dmc-benchmarks/nqfi11ij',
            'arnolds/cherry-dmc-benchmarks/yl7b73d8',
            'arnolds/cherry-dmc-benchmarks/0wp3pon1',
            'arnolds/cherry-dmc-benchmarks/i35ksnl2',
            'arnolds/cherry-dmc-benchmarks/rufodq0p',
        ],
    },
    'drqv2': {
        'ball_in_cup-catch': [
            'arnolds/cherry-dmc-benchmarks/5td1lwif',
            'arnolds/cherry-dmc-benchmarks/r30696sw',
            'arnolds/cherry-dmc-benchmarks/3g6cjmzp',
            'arnolds/cherry-dmc-benchmarks/9lzrafh7',
            'arnolds/cherry-dmc-benchmarks/9cxu9cdw',
        ],
        'cartpole-swingup': [
            'arnolds/cherry-dmc-benchmarks/5mbo4eln',
            'arnolds/cherry-dmc-benchmarks/ho4x7c5a',
            'arnolds/cherry-dmc-benchmarks/6kxkfef0',
            'arnolds/cherry-dmc-benchmarks/mdolps3d',
            'arnolds/cherry-dmc-benchmarks/0emuw4gt',
        ],
        'reacher-easy': [
            'arnolds/cherry-dmc-benchmarks/9268byco',
            'arnolds/cherry-dmc-benchmarks/txxl8zip',
            'arnolds/cherry-dmc-benchmarks/fd0vww25',
            'arnolds/cherry-dmc-benchmarks/9rgre9bc',
            'arnolds/cherry-dmc-benchmarks/a13giqvf',
        ],
        'finger-spin': [
            'arnolds/cherry-dmc-benchmarks/28ek34a6',
            'arnolds/cherry-dmc-benchmarks/s7fqxxf1',
            'arnolds/cherry-dmc-benchmarks/38pom3cn',
            'arnolds/cherry-dmc-benchmarks/q21rrctp',
            'arnolds/cherry-dmc-benchmarks/q21rrctp',
        ],
        'cheetah-run': [
            'arnolds/cherry-dmc-benchmarks/3d4eeocw',
            'arnolds/cherry-dmc-benchmarks/tq5jeomw',
            'arnolds/cherry-dmc-benchmarks/96ko3drp',
            'arnolds/cherry-dmc-benchmarks/roe4l7gu',
            'arnolds/cherry-dmc-benchmarks/roe4l7gu',
        ],
        'walker-walk': [
            'arnolds/cherry-dmc-benchmarks/dxr7rqoo',
            'arnolds/cherry-dmc-benchmarks/t69omrv5',
            'arnolds/cherry-dmc-benchmarks/fv4rqfxo',
            'arnolds/cherry-dmc-benchmarks/6i42fwtx',
            'arnolds/cherry-dmc-benchmarks/utlnxj9e',
        ],
    },
}

colors = {
    'sac': pl.Vibrant['cyan'],
    'drq': pl.Vibrant['magenta'],
    'drqv2': pl.Vibrant['teal'],
}

markers = {
    'sac': pl.markers.circle,
    'drq': pl.markers.square,
    'drqv2': pl.markers.diamond,
}

labels = {
    'sac': 'SAC',  # (proprio)
    'drq': 'DrQ',  # (vision)
    'drqv2': 'DrQ-v2',  # (vision)
}


def main():
    grid_plot = pl.Container(rows=2, cols=3, height=2*2600.0, width=3*2600.0)
    for i, task in enumerate([
        'ball_in_cup-catch',
        'cartpole-swingup',
        'reacher-easy',
        'finger-spin',
        'cheetah-run',
        'walker-walk',
    ]):
        task_domain, task_name = task.split('-')
        task_domain = task_domain.replace('_', ' ').capitalize()
        task_name = task_name.replace('_', ' ').capitalize()
        config = {
            'type': pl.ModernPlot,
            'title': f'{task_domain}: {task_name}',
            'subtitle': r'DMC',
            'xtitle': 'Steps',
            'ytitle': 'Rewards',
            'xlims': (0, 200_000),
            'ylims': (0.0, 1000.0),
            'y_notation': 'decimal',
            'legend': {
                'title': 'Algorithm',
                'inset': True,
                'loc': 'best',
                'show': True,
            },
            'results': [],
        }
        for algorithm in [
            'sac',
            'drq',
            'drqv2',
        ]:
            config['results'].append({
                'wandb_id': wandb_runs[algorithm][task],
                'x_key': 'eval_step',
                'y_key': 'test_rewards',
                'label': labels[algorithm],
                'color': colors[algorithm],
                'marker': markers[algorithm],
                'linewidth': 1.8,
                'smooth_temperature': 2.0,
                'markevery': 32,
                'samples': 4196,
                'shade': 'ci95',
            })
        plot = pl.wandb_plots.wandb_plot(config)

        # save individual figures
        # plot.save(f'results/{task}.png')

        # modify for grid figure
        plot.set_subtitle('')
        if i < 3:
            plot.set_axis(x='')
        if i in [1, 2, 4, 5]:
            plot.set_axis(y='')
        if i != 5:
            plot.set_legend(show=False)
        grid_plot.set_plot(i // 3, i % 3, plot)
    grid_plot.save('results/all.png')


if __name__ == "__main__":
    main()
