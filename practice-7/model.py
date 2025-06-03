import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
from io import BytesIO
from typing import List
from config import (COLORS, STATES, DEFAULT_GRID_SIZE,
                    DEFAULT_P_INFECT, DEFAULT_T_RECOVER)


class EpidemicModel:
    def __init__(self, size: int = DEFAULT_GRID_SIZE,
                 p_infect: float = DEFAULT_P_INFECT,
                 t_recover: int = DEFAULT_T_RECOVER):
        self.size = size
        self.p_infect = p_infect
        self.t_recover = t_recover
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.infection_time = np.zeros((self.size, self.size), dtype=int)

        center = self.size // 2
        self.grid[center-2:center+3, center-2:center+3] = STATES['INFECTED']

    def get_neighbors(self, i: int, j: int) -> List[tuple]:
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    neighbors.append((ni, nj))
        return neighbors

    def step(self):
        new_grid = self.grid.copy()
        new_infection_time = self.infection_time.copy()

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == STATES['SUSCEPTIBLE']:
                    infected_neighbors = 0
                    for ni, nj in self.get_neighbors(i, j):
                        if self.grid[ni, nj] == STATES['INFECTED']:
                            infected_neighbors += 1

                    if (infected_neighbors > 0 and
                            np.random.random() < self.p_infect):
                        new_grid[i, j] = STATES['INFECTED']
                        new_infection_time[i, j] = 0

                elif self.grid[i, j] == STATES['INFECTED']:
                    new_infection_time[i, j] = self.infection_time[i, j] + 1

                    if self.infection_time[i, j] >= self.t_recover:
                        new_grid[i, j] = STATES['RECOVERED']

        self.grid = new_grid
        self.infection_time = new_infection_time

    def get_stats(self):
        susceptible = np.sum(self.grid == STATES['SUSCEPTIBLE'])
        infected = np.sum(self.grid == STATES['INFECTED'])
        recovered = np.sum(self.grid == STATES['RECOVERED'])
        return {
            "susceptible": int(susceptible),
            "infected": int(infected),
            "recovered": int(recovered),
            "total": self.size * self.size
        }

    def to_image_base64(self) -> str:
        colors = [COLORS['susceptible'], COLORS['infected'],
                  COLORS['recovered']]
        cmap = mcolors.ListedColormap(colors)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.grid, cmap=cmap, vmin=0, vmax=2)
        ax.set_title('Модель епідемії (SIR)', fontsize=16, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['susceptible'],
                          label='Сприйнятливі (S)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['infected'],
                          label='Інфіковані (I)'),
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['recovered'],
                          label='Одужалі (R)')
        ]
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return image_base64
