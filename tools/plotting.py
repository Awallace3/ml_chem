import matplotlib.pyplot as plt


def plot_target_vs_predicted(E_model, E_target, output="plots/t.png"):
    x = [i - 3200 for i in range(3500)]
    fig = plt.figure(dpi=400)
    plt.plot(E_target, E_model, 'r.', linewidth=0.1)
    plt.plot(x, x, 'k')
    plt.xlabel('Target Energy')
    plt.ylabel('Predicted Energy')
    # plt.xlim([-3000, 0])
    # plt.ylim([-3000, 0])
    plt.savefig(output)
