import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import warnings
import time

warnings.filterwarnings('ignore')

import matplotlib

matplotlib.use('TkAgg')  # 使用TkAgg后端

# ====================== 元胞自动机核心 ======================
class AlzheimerCA:
    """阿尔茨海默病元胞自动机"""

    def __init__(self, size=50, pathology_type="amyloid_tau", treatment_type="natural"):
        self.size = size
        self.pathology_type = pathology_type
        self.treatment_type = treatment_type

        # 元胞状态: 0=健康, 1=损伤, 2=死亡
        self.grid = np.zeros((size, size), dtype=int)

        # 病理因子网格
        self.beta_grid = np.zeros((size, size))
        self.tau_grid = np.zeros((size, size))
        self.vascular_grid = np.zeros((size, size))

        # 初始化脑区
        self.init_brain_regions()

        # 初始化病理
        self.init_pathology()

        # 元胞自动机参数
        self.params = self.get_parameters()

        # 记录状态
        self.steps = 0
        self.cognitive_scores = []
        self.healthy_counts = []
        self.dead_counts = []

    def init_brain_regions(self):
        """初始化脑区元胞"""
        self.hippocampus_mask = np.zeros((self.size, self.size))
        self.entorhinal_mask = np.zeros((self.size, self.size))

        # 海马体
        center_x, center_y = self.size // 3, self.size // 2
        for i in range(self.size):
            for j in range(self.size):
                if ((i - center_x) ** 2 / 9 + (j - center_y) ** 2) < 25:
                    self.hippocampus_mask[i, j] = 1

        # 内嗅皮层
        center_x, center_y = 2 * self.size // 3, self.size // 2
        for i in range(self.size):
            for j in range(self.size):
                if ((i - center_x) ** 2 / 16 + (j - center_y) ** 2) < 20:
                    self.entorhinal_mask[i, j] = 1

    def init_pathology(self):
        """初始化病理元胞"""
        np.random.seed(42)

        if self.pathology_type == "amyloid_tau":
            for i in range(self.size):
                for j in range(self.size):
                    if self.entorhinal_mask[i, j] > 0.5 and np.random.random() < 0.3:
                        self.tau_grid[i, j] = 0.4 + np.random.random() * 0.3
                        self.beta_grid[i, j] = 0.2 + np.random.random() * 0.2

        elif self.pathology_type == "vascular":
            for _ in range(6):
                cx, cy = np.random.randint(10, self.size - 10, 2)
                radius = np.random.randint(3, 6)
                for i in range(max(0, cx - radius), min(self.size, cx + radius + 1)):
                    for j in range(max(0, cy - radius), min(self.size, cy + radius + 1)):
                        dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                        if dist < radius:
                            self.vascular_grid[i, j] = 0.5 + np.random.random() * 0.3

        elif self.pathology_type == "mixed":
            for i in range(self.size):
                for j in range(self.size):
                    if np.random.random() < 0.2 and self.entorhinal_mask[i, j] > 0.5:
                        self.tau_grid[i, j] = 0.3 + np.random.random() * 0.3
                        self.beta_grid[i, j] = 0.2 + np.random.random() * 0.2
                    if np.random.random() < 0.2:
                        self.vascular_grid[i, j] = 0.3 + np.random.random() * 0.2

    def get_parameters(self):
        """获取元胞自动机参数"""
        params = {
            "amyloid_tau": {"beta_spread": 0.02, "tau_spread": 0.025, "generation_rate": 0.15},
            "vascular": {"beta_spread": 0.01, "tau_spread": 0.015, "generation_rate": 0.1},
            "mixed": {"beta_spread": 0.025, "tau_spread": 0.03, "generation_rate": 0.25}
        }
        return params.get(self.pathology_type, params["amyloid_tau"])

    def get_treatment_factor(self):
        """获取治疗因子"""
        treatments = {
            "natural": {"beta": 1.0, "tau": 1.0, "vascular": 1.0},
            "amyloid_inhibitor": {"beta": 0.7, "tau": 0.9, "vascular": 0.9},
            "combination": {"beta": 0.6, "tau": 0.5, "vascular": 0.7}
        }
        return treatments.get(self.treatment_type, treatments["natural"])

    def get_neighbors(self, i, j, radius=1):
        """获取邻域元胞"""
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                ni, nj = i + dx, j + dy
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    neighbors.append((ni, nj))
        return neighbors

    def calculate_damage(self, i, j):
        """计算元胞总损伤"""
        beta_damage = self.beta_grid[i, j] * 0.4
        tau_damage = self.tau_grid[i, j] * 0.5
        vascular_damage = self.vascular_grid[i, j] * 0.3

        vulnerability = 1.0
        if self.hippocampus_mask[i, j] > 0.5:
            vulnerability = 1.5
        elif self.entorhinal_mask[i, j] > 0.5:
            vulnerability = 1.3

        total_damage = (beta_damage + tau_damage + vascular_damage) * vulnerability

        return np.clip(total_damage, 0, 1.0)

    def update_grid(self):
        """更新整个元胞自动机网格"""
        new_grid = self.grid.copy()
        new_beta = self.beta_grid.copy()
        new_tau = self.tau_grid.copy()
        new_vascular = self.vascular_grid.copy()

        treatment = self.get_treatment_factor()

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 2:
                    continue

                # 病理传播
                neighbors = self.get_neighbors(i, j, 1)
                for ni, nj in neighbors:
                    spread_factor = 0.8 + np.random.random() * 0.4
                    new_beta[i, j] += self.beta_grid[ni, nj] * 0.015 * spread_factor
                    new_tau[i, j] += self.tau_grid[ni, nj] * 0.02 * spread_factor
                    new_vascular[i, j] += self.vascular_grid[ni, nj] * 0.008

                # 病理生成
                if self.entorhinal_mask[i, j] > 0.5 and np.random.random() < self.params["generation_rate"]:
                    new_tau[i, j] += 0.05 + np.random.random() * 0.03
                    new_beta[i, j] += 0.03 + np.random.random() * 0.02

                # 血管损伤加重
                if new_vascular[i, j] > 0.3:
                    new_beta[i, j] += 0.01
                    new_tau[i, j] += 0.015

                # 应用治疗
                new_beta[i, j] *= treatment["beta"]
                new_tau[i, j] *= treatment["tau"]
                new_vascular[i, j] *= treatment["vascular"]

                # 限制值范围
                new_beta[i, j] = np.clip(new_beta[i, j], 0, 1)
                new_tau[i, j] = np.clip(new_tau[i, j], 0, 1)
                new_vascular[i, j] = np.clip(new_vascular[i, j], 0, 1)

                # 计算总损伤
                total_damage = self.calculate_damage(i, j)

                # 神经保护
                if self.grid[i, j] == 1 and self.treatment_type != "natural":
                    if np.random.random() < 0.1:
                        new_grid[i, j] = 0
                        continue

                # 更新元胞状态
                if total_damage < 0.3:
                    new_grid[i, j] = 0
                elif total_damage < 0.7:
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 2

        # 更新网格
        self.grid = new_grid
        self.beta_grid = new_beta
        self.tau_grid = new_tau
        self.vascular_grid = new_vascular

        # 记录状态
        self.steps += 1
        self.cognitive_scores.append(self.calculate_cognitive_score())
        self.healthy_counts.append(np.sum(self.grid == 0))
        self.dead_counts.append(np.sum(self.grid == 2))

    def calculate_cognitive_score(self):
        """计算认知分数"""
        total_cells = self.size * self.size
        healthy_cells = np.sum(self.grid == 0)
        damaged_cells = np.sum(self.grid == 1)

        base_score = (healthy_cells + damaged_cells * 0.3) / total_cells

        hippocampus_cells = np.sum(self.hippocampus_mask > 0.5)
        if hippocampus_cells > 0:
            hippocampus_healthy = np.sum((self.grid == 0) & (self.hippocampus_mask > 0.5))
            hippocampus_score = hippocampus_healthy / hippocampus_cells
        else:
            hippocampus_score = 0

        cognitive_score = 0.7 * base_score + 0.3 * hippocampus_score

        return np.clip(cognitive_score, 0, 1)

    def get_statistics(self):
        """获取统计信息"""
        total = self.size * self.size
        stats = {
            "step": self.steps,
            "healthy": np.sum(self.grid == 0),
            "damaged": np.sum(self.grid == 1),
            "dead": np.sum(self.grid == 2),
            "cognitive_score": self.cognitive_scores[-1] if self.cognitive_scores else 0,
            "beta_level": np.mean(self.beta_grid),
            "tau_level": np.mean(self.tau_grid)
        }
        return stats


# ====================== 对比图1：不同病理机制对比 ======================
def show_pathology_comparison():
    """显示不同病理机制对比图"""
    print("\n" + "=" * 60)
    print("对比图1: 不同病理机制对比")
    print("=" * 60)

    # 创建3种病理机制的元胞自动机
    pathologies = ["amyloid_tau", "vascular", "mixed"]
    pathology_names = {
        "amyloid_tau": "Classic AD\n(β-amyloid + Tau)",
        "vascular": "Vascular Dementia\n(Vascular Damage)",
        "mixed": "Mixed Pathology\n(AD + Vascular)"
    }

    cas = [AlzheimerCA(size=50, pathology_type=pt, treatment_type="natural") for pt in pathologies]

    # 创建图形
    fig = plt.figure(figsize=(16, 12))  # 增加图形高度
    fig.suptitle("Comparison 1: Impact of Different Pathology Mechanisms",
                 fontsize=16, fontweight='bold', y=0.97)

    # 提前定义网格布局
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1.2])

    # 存储子图引用
    axes_neurons = []
    axes_beta = []
    axes_tau = []

    # 颜色映射
    cmap_neurons = LinearSegmentedColormap.from_list('neurons', ['#2ecc71', '#f39c12', '#e74c3c'], N=3)
    cmap_beta = LinearSegmentedColormap.from_list('beta', ['#ffffff', '#3498db', '#2980b9'], N=256)
    cmap_tau = LinearSegmentedColormap.from_list('tau', ['#ffffff', '#27ae60', '#229954'], N=256)

    # 创建动画函数
    def animate(frame):
        if frame >= 100:  # 减少到100步
            ani.event_source.stop()
            plt.close(fig)  # 关闭图形
            return

        for ca in cas:
            ca.update_grid()

        # 清除所有子图
        for ax_list in [axes_neurons, axes_beta, axes_tau]:
            for ax in ax_list:
                ax.clear()
        if hasattr(animate, 'ax_cog'):
            animate.ax_cog.clear()

        # 第一行：神经元状态
        for i, (ca, pt) in enumerate(zip(cas, pathologies)):
            if i < len(axes_neurons):
                ax = axes_neurons[i]
            else:
                ax = fig.add_subplot(gs[0, i])
                axes_neurons.append(ax)

            im = ax.imshow(ca.grid, cmap=cmap_neurons, vmin=0, vmax=2,
                           interpolation='nearest', aspect='equal')
            ax.set_title(f"{pathology_names[pt]}\nStep: {ca.steps}", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # 添加脑区标记
            center_x, center_y = 50 // 3, 50 // 2
            hippo = plt.Circle((center_y, center_x), 4, fill=False,
                               color='blue', linewidth=1.5, alpha=0.6)
            ax.add_patch(hippo)

            center_x, center_y = 2 * 50 // 3, 50 // 2
            ento = plt.Circle((center_y, center_x), 3, fill=False,
                              color='red', linewidth=1.5, alpha=0.6)
            ax.add_patch(ento)

            if i == 0 and not hasattr(animate, 'colorbar_added'):
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                    shrink=0.8, pad=0.05)
                cbar.set_ticks([0, 1, 2])
                cbar.set_ticklabels(['Healthy', 'Damaged', 'Dead'])
                cbar.ax.tick_params(labelsize=7)
                animate.colorbar_added = True

        # 第二行：β淀粉样蛋白分布
        for i, ca in enumerate(cas):
            if i < len(axes_beta):
                ax = axes_beta[i]
            else:
                ax = fig.add_subplot(gs[1, i])
                axes_beta.append(ax)

            im = ax.imshow(ca.beta_grid, cmap=cmap_beta, vmin=0, vmax=1,
                           interpolation='nearest', aspect='equal')
            ax.set_title("β-amyloid Distribution", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # 第三行：Tau蛋白分布
        for i, ca in enumerate(cas):
            if i < len(axes_tau):
                ax = axes_tau[i]
            else:
                ax = fig.add_subplot(gs[2, i])
                axes_tau.append(ax)

            im = ax.imshow(ca.tau_grid, cmap=cmap_tau, vmin=0, vmax=1,
                           interpolation='nearest', aspect='equal')
            ax.set_title("Tau Protein Distribution", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0 and not hasattr(animate, 'tau_colorbar_added'):
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                    shrink=0.8, pad=0.05)
                cbar.ax.tick_params(labelsize=7)
                animate.tau_colorbar_added = True

        # 第四行：认知功能曲线（单独一行，不重叠）
        if not hasattr(animate, 'ax_cog'):
            animate.ax_cog = fig.add_subplot(gs[3, :])
        ax_cog = animate.ax_cog

        colors = ['#e74c3c', '#3498db', '#2ecc71']

        for i, (ca, pt_name) in enumerate(zip(cas, pathologies)):
            if len(ca.cognitive_scores) > 0:
                x = list(range(len(ca.cognitive_scores)))
                ax_cog.plot(x, ca.cognitive_scores, color=colors[i], linewidth=2,
                            label=pathology_names[pt_name].split('\n')[0], alpha=0.8)

        ax_cog.set_title("Cognitive Function Over Time", fontsize=12, fontweight='bold', pad=10)
        ax_cog.set_xlabel("Time Steps", fontsize=10)
        ax_cog.set_ylabel("Cognitive Score", fontsize=10)
        ax_cog.grid(True, alpha=0.3, linestyle='--')
        ax_cog.set_ylim(0, 1.1)
        ax_cog.set_xlim(0, 100)

        # 调整图例位置
        ax_cog.legend(fontsize=9, loc='upper right')

        if len(cas[0].cognitive_scores) > 0:
            current_step = len(cas[0].cognitive_scores) - 1
            ax_cog.axvline(x=current_step, color='gray', linestyle=':', alpha=0.5)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.canvas.draw()

        return []

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=100, interval=200, repeat=False, blit=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 显示统计结果
    print("\n病理机制对比结果:")
    print("-" * 50)
    for i, (ca, pt) in enumerate(zip(cas, pathologies)):
        stats = ca.get_statistics()
        name = pathology_names[pt].split('\n')[0]
        print(f"{name}:")
        print(f"  最终认知分数: {stats['cognitive_score']:.3f}")
        print(f"  健康神经元比例: {stats['healthy'] / (50 * 50) * 100:.1f}%")
        print(f"  死亡神经元比例: {stats['dead'] / (50 * 50) * 100:.1f}%")
    print("-" * 50)


# ====================== 对比图2：不同治疗方法对比 ======================
def show_treatment_comparison():
    """显示不同治疗方法对比图"""
    print("\n" + "=" * 60)
    print("对比图2: 不同治疗方法效果对比")
    print("=" * 60)

    # 创建3种治疗方法的元胞自动机
    treatments = ["natural", "amyloid_inhibitor", "combination"]
    treatment_names = {
        "natural": "No Treatment\n(Natural Progression)",
        "amyloid_inhibitor": "β-amyloid\nInhibitor",
        "combination": "Combination\nTherapy"
    }

    cas = [AlzheimerCA(size=50, pathology_type="amyloid_tau", treatment_type=t) for t in treatments]

    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Comparison 2: Effects of Different Treatment Approaches",
                 fontsize=16, fontweight='bold', y=0.97)

    # 提前定义网格布局
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1.2])

    # 存储子图引用
    axes_neurons = []
    axes_beta = []
    axes_tau = []

    # 颜色映射
    cmap_neurons = LinearSegmentedColormap.from_list('neurons', ['#2ecc71', '#f39c12', '#e74c3c'], N=3)
    cmap_beta = LinearSegmentedColormap.from_list('beta', ['#ffffff', '#3498db', '#2980b9'], N=256)
    cmap_tau = LinearSegmentedColormap.from_list('tau', ['#ffffff', '#27ae60', '#229954'], N=256)

    # 创建动画函数
    def animate(frame):
        if frame >= 100:
            ani.event_source.stop()
            plt.close(fig)
            return

        for ca in cas:
            ca.update_grid()

        # 清除所有子图
        for ax_list in [axes_neurons, axes_beta, axes_tau]:
            for ax in ax_list:
                ax.clear()
        if hasattr(animate, 'ax_cog'):
            animate.ax_cog.clear()

        # 第一行：神经元状态
        for i, (ca, t) in enumerate(zip(cas, treatments)):
            if i < len(axes_neurons):
                ax = axes_neurons[i]
            else:
                ax = fig.add_subplot(gs[0, i])
                axes_neurons.append(ax)

            im = ax.imshow(ca.grid, cmap=cmap_neurons, vmin=0, vmax=2,
                           interpolation='nearest', aspect='equal')
            ax.set_title(f"{treatment_names[t]}\nStep: {ca.steps}", fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # 添加脑区标记
            center_x, center_y = 50 // 3, 50 // 2
            hippo = plt.Circle((center_y, center_x), 4, fill=False,
                               color='blue', linewidth=1.5, alpha=0.6)
            ax.add_patch(hippo)

            center_x, center_y = 2 * 50 // 3, 50 // 2
            ento = plt.Circle((center_y, center_x), 3, fill=False,
                              color='red', linewidth=1.5, alpha=0.6)
            ax.add_patch(ento)

            if i == 0 and not hasattr(animate, 'colorbar_added'):
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                    shrink=0.8, pad=0.05)
                cbar.set_ticks([0, 1, 2])
                cbar.set_ticklabels(['Healthy', 'Damaged', 'Dead'])
                cbar.ax.tick_params(labelsize=7)
                animate.colorbar_added = True

        # 第二行：β淀粉样蛋白分布
        for i, ca in enumerate(cas):
            if i < len(axes_beta):
                ax = axes_beta[i]
            else:
                ax = fig.add_subplot(gs[1, i])
                axes_beta.append(ax)

            im = ax.imshow(ca.beta_grid, cmap=cmap_beta, vmin=0, vmax=1,
                           interpolation='nearest', aspect='equal')
            ax.set_title("β-amyloid Distribution", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # 第三行：Tau蛋白分布
        for i, ca in enumerate(cas):
            if i < len(axes_tau):
                ax = axes_tau[i]
            else:
                ax = fig.add_subplot(gs[2, i])
                axes_tau.append(ax)

            im = ax.imshow(ca.tau_grid, cmap=cmap_tau, vmin=0, vmax=1,
                           interpolation='nearest', aspect='equal')
            ax.set_title("Tau Protein Distribution", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0 and not hasattr(animate, 'tau_colorbar_added'):
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                                    shrink=0.8, pad=0.05)
                cbar.ax.tick_params(labelsize=7)
                animate.tau_colorbar_added = True

        # 第四行：认知功能曲线
        if not hasattr(animate, 'ax_cog'):
            animate.ax_cog = fig.add_subplot(gs[3, :])
        ax_cog = animate.ax_cog

        colors = ['#e74c3c', '#3498db', '#9b59b6']

        for i, (ca, t_name) in enumerate(zip(cas, treatments)):
            if len(ca.cognitive_scores) > 0:
                x = list(range(len(ca.cognitive_scores)))
                ax_cog.plot(x, ca.cognitive_scores, color=colors[i], linewidth=2,
                            label=treatment_names[t_name].split('\n')[0], alpha=0.8)

        ax_cog.set_title("Cognitive Function with Different Treatments",
                         fontsize=12, fontweight='bold', pad=10)
        ax_cog.set_xlabel("Time Steps", fontsize=10)
        ax_cog.set_ylabel("Cognitive Score", fontsize=10)
        ax_cog.grid(True, alpha=0.3, linestyle='--')
        ax_cog.set_ylim(0, 1.1)
        ax_cog.set_xlim(0, 100)

        # 调整图例位置
        ax_cog.legend(fontsize=9, loc='upper right')

        if len(cas[0].cognitive_scores) > 0:
            current_step = len(cas[0].cognitive_scores) - 1
            ax_cog.axvline(x=current_step, color='gray', linestyle=':', alpha=0.5)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.canvas.draw()

        return []

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=100, interval=200, repeat=False, blit=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 显示统计结果
    print("\n治疗方法对比结果:")
    print("-" * 50)
    for i, (ca, t) in enumerate(zip(cas, treatments)):
        stats = ca.get_statistics()
        name = treatment_names[t].split('\n')[0]
        print(f"{name}:")
        print(f"  最终认知分数: {stats['cognitive_score']:.3f}")
        print(f"  健康神经元比例: {stats['healthy'] / (50 * 50) * 100:.1f}%")
        print(f"  β淀粉样蛋白水平: {stats['beta_level']:.3f}")
        print(f"  Tau蛋白水平: {stats['tau_level']:.3f}")
    print("-" * 50)


# ====================== 对比图3：混合病理动态演化 ======================
def show_mixed_pathology_dynamics():
    """显示混合病理动态演化图"""
    print("\n" + "=" * 60)
    print("对比图3: 混合病理动态演化过程")
    print("=" * 60)

    # 创建混合病理的元胞自动机
    ca = AlzheimerCA(size=50, pathology_type="mixed", treatment_type="natural")

    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Comparison 3: Dynamic Evolution of Mixed Pathology",
                 fontsize=16, fontweight='bold', y=0.97)

    # 提前定义网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1.2])

    # 存储子图引用
    axes = {}

    # 颜色映射
    cmap_neurons = LinearSegmentedColormap.from_list('neurons', ['#2ecc71', '#f39c12', '#e74c3c'], N=3)
    cmap_beta = LinearSegmentedColormap.from_list('beta', ['#ffffff', '#3498db', '#2980b9'], N=256)
    cmap_tau = LinearSegmentedColormap.from_list('tau', ['#ffffff', '#27ae60', '#229954'], N=256)
    cmap_vascular = LinearSegmentedColormap.from_list('vascular', ['#ffffff', '#f39c12', '#d35400'], N=256)

    # 存储历史数据
    beta_history = []
    tau_history = []
    vascular_history = []
    healthy_history = []

    # 创建动画函数
    def animate(frame):
        if frame >= 80:  # 减少到80步
            ani.event_source.stop()
            plt.close(fig)
            return

        ca.update_grid()

        # 记录历史
        beta_history.append(np.mean(ca.beta_grid))
        tau_history.append(np.mean(ca.tau_grid))
        vascular_history.append(np.mean(ca.vascular_grid))
        healthy_history.append(np.sum(ca.grid == 0) / (50 * 50))

        # 清除所有子图
        for ax in axes.values():
            ax.clear()
        if hasattr(animate, 'ax_curve'):
            animate.ax_curve.clear()

        # 第一行左：神经元状态
        if 'ax1' not in axes:
            axes['ax1'] = fig.add_subplot(gs[0, 0])
        ax1 = axes['ax1']
        im1 = ax1.imshow(ca.grid, cmap=cmap_neurons, vmin=0, vmax=2,
                         interpolation='nearest', aspect='equal')
        ax1.set_title(f"Neuron States\nStep: {ca.steps}", fontsize=11, fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])

        if not hasattr(animate, 'colorbar_added'):
            cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal',
                                 shrink=0.8, pad=0.05)
            cbar1.set_ticks([0, 1, 2])
            cbar1.set_ticklabels(['Healthy', 'Damaged', 'Dead'])
            cbar1.ax.tick_params(labelsize=7)
            animate.colorbar_added = True

        # 添加脑区标记
        center_x, center_y = 50 // 3, 50 // 2
        hippo = plt.Circle((center_y, center_x), 4, fill=False,
                           color='blue', linewidth=1.5, alpha=0.6)
        ax1.add_patch(hippo)
        ax1.text(center_y, center_x - 5, 'Hippocampus',
                 color='blue', fontsize=8, ha='center', alpha=0.8)

        center_x, center_y = 2 * 50 // 3, 50 // 2
        ento = plt.Circle((center_y, center_x), 3, fill=False,
                          color='red', linewidth=1.5, alpha=0.6)
        ax1.add_patch(ento)
        ax1.text(center_y, center_x - 5, 'Entorhinal',
                 color='red', fontsize=8, ha='center', alpha=0.8)

        # 第一行中：β淀粉样蛋白
        if 'ax2' not in axes:
            axes['ax2'] = fig.add_subplot(gs[0, 1])
        ax2 = axes['ax2']
        im2 = ax2.imshow(ca.beta_grid, cmap=cmap_beta, vmin=0, vmax=1,
                         interpolation='nearest', aspect='equal')
        ax2.set_title("β-amyloid Distribution", fontsize=11, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])

        if not hasattr(animate, 'beta_colorbar_added'):
            plt.colorbar(im2, ax=ax2, orientation='horizontal', shrink=0.8, pad=0.05)
            animate.beta_colorbar_added = True

        # 第一行右：Tau蛋白
        if 'ax3' not in axes:
            axes['ax3'] = fig.add_subplot(gs[0, 2])
        ax3 = axes['ax3']
        im3 = ax3.imshow(ca.tau_grid, cmap=cmap_tau, vmin=0, vmax=1,
                         interpolation='nearest', aspect='equal')
        ax3.set_title("Tau Protein Distribution", fontsize=11, fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])

        if not hasattr(animate, 'tau_colorbar_added'):
            plt.colorbar(im3, ax=ax3, orientation='horizontal', shrink=0.8, pad=0.05)
            animate.tau_colorbar_added = True

        # 第二行左：血管损伤
        if 'ax4' not in axes:
            axes['ax4'] = fig.add_subplot(gs[1, 0])
        ax4 = axes['ax4']
        im4 = ax4.imshow(ca.vascular_grid, cmap=cmap_vascular, vmin=0, vmax=1,
                         interpolation='nearest', aspect='equal')
        ax4.set_title("Vascular Damage", fontsize=11, fontweight='bold')
        ax4.set_xticks([])
        ax4.set_yticks([])

        if not hasattr(animate, 'vascular_colorbar_added'):
            plt.colorbar(im4, ax=ax4, orientation='horizontal', shrink=0.8, pad=0.05)
            animate.vascular_colorbar_added = True

        # 第二行中右：病理因子时间序列
        if not hasattr(animate, 'ax_curve'):
            animate.ax_curve = fig.add_subplot(gs[1:, 1:])
        ax_curve = animate.ax_curve

        if len(beta_history) > 0:
            x = list(range(len(beta_history)))
            ax_curve.plot(x, beta_history, color='#3498db', linewidth=2,
                          label='β-amyloid', alpha=0.8)
            ax_curve.plot(x, tau_history, color='#27ae60', linewidth=2,
                          label='Tau protein', alpha=0.8)
            ax_curve.plot(x, vascular_history, color='#f39c12', linewidth=2,
                          label='Vascular damage', alpha=0.8)
            ax_curve.plot(x, healthy_history, color='#2ecc71', linewidth=2,
                          label='Healthy ratio', alpha=0.8)

        ax_curve.set_title("Pathology Factors Over Time", fontsize=12, fontweight='bold', pad=10)
        ax_curve.set_xlabel("Time Steps", fontsize=10)
        ax_curve.set_ylabel("Level / Ratio", fontsize=10)
        ax_curve.grid(True, alpha=0.3, linestyle='--')
        ax_curve.set_ylim(0, 1.1)
        ax_curve.set_xlim(0, 80)

        # 调整图例位置
        ax_curve.legend(fontsize=9, loc='upper right')

        if len(beta_history) > 0:
            current_step = len(beta_history) - 1
            ax_curve.axvline(x=current_step, color='gray', linestyle=':', alpha=0.5)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.canvas.draw()

        return []

    # 创建动画
    ani = FuncAnimation(fig, animate, frames=80, interval=200, repeat=False, blit=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 显示统计结果
    stats = ca.get_statistics()
    print(f"\n混合病理演化结果 (80步后):")
    print("-" * 50)
    print(f"最终认知分数: {stats['cognitive_score']:.3f}")
    print(f"健康神经元: {stats['healthy']} ({stats['healthy'] / (50 * 50) * 100:.1f}%)")
    print(f"损伤神经元: {stats['damaged']} ({stats['damaged'] / (50 * 50) * 100:.1f}%)")
    print(f"死亡神经元: {stats['dead']} ({stats['dead'] / (50 * 50) * 100:.1f}%)")
    print(f"β淀粉样蛋白水平: {stats['beta_level']:.3f}")
    print(f"Tau蛋白水平: {stats['tau_level']:.3f}")
    print("-" * 50)


# ====================== 主程序 ======================
def main():
    """主程序"""
    print("阿尔茨海默病元胞自动机模拟系统")
    print("=" * 60)
    print("本系统将依次显示3个对比图:")
    print("1. 不同病理机制对比 (3种病理类型)")
    print("2. 不同治疗方法对比 (3种治疗方案)")
    print("3. 混合病理动态演化 (详细病理因子变化)")
    print("=" * 60)

    time.sleep(1)

    # 依次调用3个对比图函数
    try:
        show_pathology_comparison()
        time.sleep(1)

        show_treatment_comparison()
        time.sleep(1)

        show_mixed_pathology_dynamics()

    except Exception as e:
        print(f"运行时出错: {e}")
        print("尝试重新运行或检查matplotlib配置")

    print("\n" + "=" * 60)
    print("所有对比图显示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()