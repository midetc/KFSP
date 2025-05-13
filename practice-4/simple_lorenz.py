import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def plot_lorenz():
    t_span = (0, 50)
    initial_state = [1.0, 1.0, 1.0]
    perturbed_state = [1.0001, 1.0, 1.0]
    
    t = np.linspace(t_span[0], t_span[1], 5000)
    sol1 = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t)
    sol2 = solve_ivp(lorenz_system, t_span, perturbed_state, t_eval=t)
    
    states1 = sol1.y.T
    states2 = sol2.y.T
    
    fig1 = plt.figure(figsize=(10, 8))
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot(states1[:, 0], states1[:, 1], states1[:, 2], lw=0.5, color='blue')
    ax.scatter(states1[0, 0], states1[0, 1], states1[0, 2], color='red', s=40, label='Початок')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Атрактор Лоренца")
    plt.savefig('lorenz.png', dpi=300)
    plt.show()
    
    distances = np.sqrt(np.sum((states1 - states2)**2, axis=1))
    
    fig2, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(t, states1[:, 0], 'b-', label='Початкова')
    axs[0].plot(t, states2[:, 0], 'r-', label='З малим збуренням')
    axs[0].set_xlabel('Час')
    axs[0].set_ylabel('X координата')
    axs[0].set_title('Розходження траєкторій з часом')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].semilogy(t, distances)
    axs[1].set_xlabel('Час')
    axs[1].set_ylabel('Відстань (лог. шкала)')
    axs[1].set_title('Експоненційне розходження (ефект метелика)')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_lorenz() 