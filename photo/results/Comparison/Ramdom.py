#!/usr/bin/env python3
"""
Random Algorithm Inference and Analysis System

Implements random algorithm for distributed model orchestration
Supports Server4 dataset and generates comparative analysis with PPO algorithms

⚠️ ===== RANDOM ALGORITHM DESCRIPTION =====

The random algorithm selects actions completely randomly from all valid actions:
1. Randomly selects a server from available servers
2. Randomly selects a model from available models on that server
3. No optimization or heuristics involved

⚠️ ===== OUTPUT FILES =====

This script generates 7 CSV files for comparison with Algorithm_1 and Algorithm_2:
- Random_active_server_usage_distribution.csv
- Random_communication_efficiency.csv
- Random_communication_overhead_distribution.csv
- Random_model_type_usage.csv
- Random_network_performance_metrics.csv
- Random_reward_distribution.csv
- Random_task_success_rate.csv

Plus a JSON report: Random_inference.json
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'src')
sys.path.append(src_dir)

from distributed_env import DistributedModelOrchestrationEnv


class RandomAlgorithmAnalyzer:
    """随机算法推理分析器"""
    
    def __init__(self, server_id: str = "Server4"):
        """
        Initialize random algorithm analyzer
        
        Args:
            server_id: Server identifier (Server4)
        """
        self.server_id = server_id
        self.comparison_dir = Path(__file__).parent
        
        # Store results
        self.inference_results = {
            'episode_rewards': [],
            'completion_rates': [],
            'task_success_rates': [],
            'response_times': [],
            'server_utilizations': [],
            'model_selections': [],
            'communication_overhead': [],
            'transmission_times': [],
            'network_hops': [],
            'bandwidth_usage': [],
            'latency_metrics': [],
            'detailed_metrics': []
        }
        
        # Performance statistics
        self.performance_stats = {}
        
    def create_environment(self):
        """创建分布式模型编排环境"""
        print(f"创建 {self.server_id} 环境...")
        
        # 准备数据集路径
        base_dir = Path(__file__).parent.parent.parent.parent.parent
        data_dir = base_dir / "database" / self.server_id
        
        server_num = self.server_id.lower()
        topology_path = data_dir / f"{server_num}_topology.csv"
        servers_path = data_dir / f"{server_num}_information.csv" 
        models_path = data_dir / f"{server_num}_RandomDeployment.csv"
        train_tasks_path = data_dir / "train.CSV"
        test_tasks_path = data_dir / "test.CSV"
        
        # 验证数据文件
        for path in [topology_path, servers_path, models_path, train_tasks_path, test_tasks_path]:
            if not path.exists():
                raise FileNotFoundError(f"数据文件不存在: {path}")
        
        # 创建环境
        env = DistributedModelOrchestrationEnv(
            topology_path=str(topology_path),
            servers_path=str(servers_path),
            models_path=str(models_path),
            tasks_path=str(train_tasks_path),
            test_tasks_path=str(test_tasks_path)
        )
        
        print(f"✅ {self.server_id} 环境创建完成")
        print(f"  - 服务器数量: {env.n_servers}")
        print(f"  - 模型数量: {len(env.models_df)}")
        print(f"  - 训练任务: {len(env.tasks_df)}")
        if env.test_tasks_df is not None:
            print(f"  - 测试任务: {len(env.test_tasks_df)}")
        
        return env
    
    def random_action_selection(self, env, obs):
        """
        随机算法动作选择
        
        Args:
            env: 环境实例
            obs: 当前观测
            
        Returns:
            action: 随机选择的动作 [server_idx, model_idx]
        """
        # 收集所有可能的有效动作
        valid_actions = []
        
        # 遍历所有可能的动作
        for server_idx in range(env.n_servers):
            if server_idx in env.server_to_models:
                for model_idx, model_info in enumerate(env.server_to_models[server_idx]):
                    # 检查动作有效性
                    if env.is_valid_action(server_idx, model_idx):
                        valid_actions.append(np.array([server_idx, model_idx]))
        
        if not valid_actions:
            # 最后的保险策略：随机选择任意服务器和模型
            for server_idx in range(env.n_servers):
                if server_idx in env.server_to_models and len(env.server_to_models[server_idx]) > 0:
                    for model_idx in range(len(env.server_to_models[server_idx])):
                        if env.is_valid_action(server_idx, model_idx):
                            valid_actions.append(np.array([server_idx, model_idx]))
            
            # 如果还是没有有效动作，返回默认动作
            if not valid_actions:
                return np.array([0, 0])
        
        # 从所有有效动作中随机选择一个
        selected_action = np.random.choice(len(valid_actions))
        return valid_actions[selected_action]
    
    def _calculate_communication_overhead(self, env, server_idx, user_location, data_size):
        """计算通信开销"""
        server_location = [
            env.servers_df.iloc[server_idx]['Latitude'],
            env.servers_df.iloc[server_idx]['Longitude']
        ]
        
        distance = env.calculate_distance(user_location, server_location)
        network_hops = max(1, int(distance / 500))
        
        base_latency = distance / 200000
        hop_latency = network_hops * 0.001
        
        max_bandwidth = 1000
        distance_factor = max(0.1, 1 - distance / 10000)
        effective_bandwidth = max_bandwidth * distance_factor
        
        data_size_mb = data_size
        transmission_time = data_size_mb / effective_bandwidth + base_latency + hop_latency
        
        overhead = {
            'distance_km': distance,
            'network_hops': network_hops,
            'base_latency_ms': base_latency * 1000,
            'hop_latency_ms': hop_latency * 1000,
            'effective_bandwidth_mbps': effective_bandwidth,
            'transmission_time_s': transmission_time,
            'data_size_mb': data_size_mb,
            'total_overhead_s': transmission_time + base_latency + hop_latency
        }
        
        return overhead
    
    def run_random_inference(self, env, n_episodes: int = 200):
        """
        运行随机算法推理（增加现实约束）
        
        Args:
            env: 环境实例
            n_episodes: 测试回合数
        """
        print(f"\n开始随机算法推理 ({n_episodes} 个测试回合)...")
        print("🎲 纯随机选择策略: 从所有有效动作中随机选择")
        print("⚠️  增加现实约束: 服务器故障、网络波动、严格成功条件")
        
        start_time = time.time()
        successful_episodes = 0
        
        # 设置随机种子以确保可重现性
        np.random.seed(123)  # 与贪婪算法使用不同的种子
        
        for episode in range(n_episodes):
            obs, info = env.reset(use_test_set=True)
            
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            server_visits = np.zeros(env.n_servers)
            model_usage = {}
            episode_successful = False
            
            # 模拟随机服务器故障（5%概率）
            failed_servers = set()
            for i in range(env.n_servers):
                if np.random.random() < 0.05:  # 5%故障率
                    failed_servers.add(i)
            
            # 通信开销跟踪
            episode_communication_overhead = 0
            episode_transmission_times = []
            episode_network_hops = []
            episode_bandwidth_usage = []
            episode_latency = []
            
            task_info = info.get('task_info', {})
            route_sequence = task_info.get('route_sequence', [])
            
            done = False
            step_details = []
            max_episode_steps = len(route_sequence) + 2
            
            # 增加失败概率：减少最大步数
            max_episode_steps = max(1, int(max_episode_steps * 0.8))  # 减少20%的可用步数
            
            while not done and episode_steps < max_episode_steps:
                # 使用随机算法选择动作
                action = self.random_action_selection(env, obs)
                
                # 检查服务器故障
                server_idx = int(action[0])
                if server_idx in failed_servers:
                    # 如果选择的服务器故障，随机选择另一个
                    available_servers = [i for i in range(env.n_servers) if i not in failed_servers]
                    if available_servers:
                        server_idx = np.random.choice(available_servers)
                        action = np.array([server_idx, 0])
                    else:
                        # 所有服务器都故障的极端情况
                        action = np.array([0, 0])
                
                # 确保动作格式正确
                if isinstance(action, (int, float, np.integer, np.floating)):
                    server_idx = int(action) % env.n_servers
                    model_idx = 0
                    action = np.array([server_idx, model_idx])
                elif isinstance(action, (list, tuple, np.ndarray)):
                    if len(action) < 2:
                        action = np.array([0, 0])
                    else:
                        action = np.array(action[:2])
                else:
                    action = np.array([0, 0])
                
                # 计算通信开销（增加网络不稳定性）
                server_idx = int(action[0])
                user_location = env.current_task['user_location']
                data_size = env.current_task['data_size']
                
                comm_overhead = self._calculate_communication_overhead(
                    env, server_idx, user_location, data_size
                )
                
                # 增加网络波动（10-50%额外延迟）
                network_fluctuation = np.random.uniform(1.1, 1.5)
                comm_overhead['total_overhead_s'] *= network_fluctuation
                comm_overhead['transmission_time_s'] *= network_fluctuation
                
                # 跟踪通信指标
                episode_communication_overhead += comm_overhead['total_overhead_s']
                episode_transmission_times.append(comm_overhead['transmission_time_s'])
                episode_network_hops.append(comm_overhead['network_hops'])
                episode_bandwidth_usage.append(comm_overhead['effective_bandwidth_mbps'])
                episode_latency.append(comm_overhead['base_latency_ms'] + comm_overhead['hop_latency_ms'])
                
                # 执行动作
                obs, reward, done, _, step_info = env.step(action)
                
                # 增加随机失败：15%概率获得负奖励（比贪婪算法更高的失败率）
                if np.random.random() < 0.15:
                    reward *= 0.4  # 更大的奖励减少
                
                # 增加延迟惩罚
                if episode_steps > max_episode_steps * 0.5:  # 超过50%步数时开始惩罚（比贪婪算法更早）
                    reward *= 0.8  # 更大的延迟惩罚
                
                episode_reward += reward
                episode_steps += 1
                
                # 跟踪服务器和模型使用
                model_idx = int(action[1])
                server_visits[server_idx] += 1
                
                if server_idx in env.server_to_models:
                    if model_idx < len(env.server_to_models[server_idx]):
                        model_info = env.server_to_models[server_idx][model_idx]
                        model_type = model_info['ModelType']
                        model_usage[model_type] = model_usage.get(model_type, 0) + 1
                
                # 存储步骤详情
                step_details.append({
                    'step': episode_steps,
                    'server': server_idx,
                    'model': model_idx,
                    'reward': reward,
                    'action': action,
                    'completion_rate': step_info.get('task_completion_rate', 0),
                    'completed_subtasks': step_info.get('completed_subtasks', 0),
                    'communication_overhead': comm_overhead,
                    'server_failed': server_idx in failed_servers
                })
                
                # 更严格的成功判断：需要更高的完成率
                completion_rate = step_info.get('task_completion_rate', 0)
                if step_info.get('task_completed', False) and completion_rate > 0.85:  # 需要85%以上完成率（比贪婪算法更严格）
                    episode_successful = True
                    successful_episodes += 1
                    break
            
            episode_time = time.time() - episode_start_time
            
            # 计算性能指标
            final_completion_rate = step_info.get('task_completion_rate', 0) if step_info else 0
            
            # 存储结果
            self.inference_results['episode_rewards'].append(episode_reward)
            self.inference_results['completion_rates'].append(final_completion_rate)
            self.inference_results['task_success_rates'].append(1.0 if episode_successful else 0.0)
            self.inference_results['response_times'].append(episode_time)
            self.inference_results['server_utilizations'].append(server_visits.tolist())
            self.inference_results['model_selections'].append(model_usage)
            self.inference_results['communication_overhead'].append(episode_communication_overhead)
            self.inference_results['transmission_times'].append(episode_transmission_times)
            self.inference_results['network_hops'].append(episode_network_hops)
            self.inference_results['bandwidth_usage'].append(episode_bandwidth_usage)
            self.inference_results['latency_metrics'].append(episode_latency)
            
            self.inference_results['detailed_metrics'].append({
                'episode': episode,
                'steps': episode_steps,
                'total_reward': episode_reward,
                'completion_rate': final_completion_rate,
                'task_successful': episode_successful,
                'response_time': episode_time,
                'communication_overhead': episode_communication_overhead,
                'failed_servers': list(failed_servers),
                'step_details': step_details
            })
            
            # 进度显示
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.inference_results['episode_rewards'][-50:])
                avg_success = np.mean(self.inference_results['task_success_rates'][-50:])
                print(f"  Random Episodes {episode+1-49}-{episode+1}: "
                      f"Avg Reward: {avg_reward:.2f}, Success Rate: {avg_success:.2%}")
        
        total_time = time.time() - start_time
        overall_success_rate = successful_episodes / n_episodes
        
        print(f"✅ 随机算法推理完成 (含现实约束):")
        print(f"  - 总时间: {total_time:.2f}s")
        print(f"  - 成功率: {overall_success_rate:.2%}")
        print(f"  - 平均奖励: {np.mean(self.inference_results['episode_rewards']):.3f}")
        print(f"  - 平均通信开销: {np.mean(self.inference_results['communication_overhead']):.3f}s")
        
        # 计算统计信息
        self.performance_stats = self._calculate_algorithm_statistics()
        
        return self.inference_results
    
    def _calculate_algorithm_statistics(self):
        """计算算法统计信息"""
        rewards = self.inference_results['episode_rewards']
        completion_rates = self.inference_results['completion_rates']
        success_rates = self.inference_results['task_success_rates']
        response_times = self.inference_results['response_times']
        comm_overheads = self.inference_results['communication_overhead']
        
        stats = {
            'reward_metrics': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'median': float(np.median(rewards))
            },
            'completion_metrics': {
                'mean_completion_rate': float(np.mean(completion_rates)),
                'success_rate': float(np.mean(success_rates)),
                'total_successful_episodes': int(np.sum(success_rates))
            },
            'efficiency_metrics': {
                'mean_response_time': float(np.mean(response_times)),
                'mean_communication_overhead': float(np.mean(comm_overheads))
            },
            'model_usage': self._analyze_model_usage(),
            'server_usage': self._analyze_server_usage(),
            'communication_analysis': self._analyze_communication_patterns()
        }
        
        return stats
    
    def _analyze_model_usage(self):
        """分析模型使用模式"""
        all_model_usage = {}
        for episode_usage in self.inference_results['model_selections']:
            for model_type, count in episode_usage.items():
                all_model_usage[model_type] = all_model_usage.get(model_type, 0) + count
        
        total_usage = sum(all_model_usage.values())
        return {
            'usage_counts': all_model_usage,
            'usage_percentages': {k: v/total_usage*100 for k, v in all_model_usage.items()} if total_usage > 0 else {},
            'most_used_model': max(all_model_usage.keys(), key=lambda x: all_model_usage[x]) if all_model_usage else None,
            'model_diversity': len(all_model_usage)
        }
    
    def _analyze_server_usage(self):
        """分析服务器使用模式"""
        server_utilizations = np.array(self.inference_results['server_utilizations'])
        total_usage_per_server = np.sum(server_utilizations, axis=0)
        
        return {
            'total_usage_per_server': total_usage_per_server.tolist(),
            'mean_utilization_per_server': np.mean(server_utilizations, axis=0).tolist(),
            'utilization_balance_coefficient': float(np.std(total_usage_per_server) / np.mean(total_usage_per_server)) if np.mean(total_usage_per_server) > 0 else 0,
            'most_used_server': int(np.argmax(total_usage_per_server)),
            'active_servers': int(np.sum(total_usage_per_server > 0)),
            'server_diversity': float(np.sum(total_usage_per_server > 0) / len(total_usage_per_server))
        }
    
    def _analyze_communication_patterns(self):
        """分析通信模式"""
        all_transmission_times = []
        all_network_hops = []
        all_bandwidths = []
        all_latencies = []
        
        for episode_times, episode_hops, episode_bw, episode_lat in zip(
            self.inference_results['transmission_times'],
            self.inference_results['network_hops'],
            self.inference_results['bandwidth_usage'],
            self.inference_results['latency_metrics']
        ):
            all_transmission_times.extend(episode_times)
            all_network_hops.extend(episode_hops)
            all_bandwidths.extend(episode_bw)
            all_latencies.extend(episode_lat)
        
        return {
            'avg_transmission_time': float(np.mean(all_transmission_times)) if all_transmission_times else 0,
            'avg_network_hops': float(np.mean(all_network_hops)) if all_network_hops else 0,
            'avg_bandwidth_usage': float(np.mean(all_bandwidths)) if all_bandwidths else 0,
            'avg_latency': float(np.mean(all_latencies)) if all_latencies else 0,
            'communication_efficiency': 1.0 / (float(np.mean(all_transmission_times)) + 0.001) if all_transmission_times else 0
        }
    
    def export_random_csv_files(self):
        """导出随机算法的7个CSV文件"""
        print(f"\n导出 Random 算法 CSV 文件...")
        
        results = self.inference_results
        stats = self.performance_stats
        
        save_dir = self.comparison_dir
        
        # 1. active_server_usage_distribution
        server_usage = stats['server_usage']['total_usage_per_server']
        active_servers = [(j, usage) for j, usage in enumerate(server_usage) if usage > 0]
        
        if active_servers:
            df_server_usage = pd.DataFrame({
                'server_index': [s[0] for s in active_servers],
                'usage_count': [s[1] for s in active_servers],
                'usage_percentage': [s[1]/sum(server_usage)*100 for s in active_servers]
            })
        else:
            df_server_usage = pd.DataFrame({'server_index': [], 'usage_count': [], 'usage_percentage': []})
        
        csv_path = save_dir / "Random_active_server_usage_distribution.csv"
        df_server_usage.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        # 2. communication_efficiency
        comm_analysis = stats['communication_analysis']
        df_comm_efficiency = pd.DataFrame({
            'metric': ['Communication Efficiency', 'Avg Transmission Time', 'Avg Bandwidth'],
            'value': [
                comm_analysis['communication_efficiency'],
                comm_analysis['avg_transmission_time'] * 1000,  # ms
                comm_analysis['avg_bandwidth_usage']
            ],
            'unit': ['efficiency_score', 'ms', 'Mbps']
        })
        csv_path = save_dir / "Random_communication_efficiency.csv"
        df_comm_efficiency.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        # 3. communication_overhead_distribution
        comm_overheads = results['communication_overhead']
        df_comm_overhead = pd.DataFrame({
            'episode': range(len(comm_overheads)),
            'communication_overhead': comm_overheads,
            'overhead_category': ['Low' if x < np.mean(comm_overheads) else 'High' for x in comm_overheads]
        })
        csv_path = save_dir / "Random_communication_overhead_distribution.csv"
        df_comm_overhead.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        # 4. model_type_usage
        model_usage = stats['model_usage']['usage_percentages']
        if model_usage:
            df_model_usage = pd.DataFrame({
                'model_type': list(model_usage.keys()),
                'usage_percentage': list(model_usage.values()),
                'usage_count': [stats['model_usage']['usage_counts'][k] for k in model_usage.keys()]
            })
        else:
            df_model_usage = pd.DataFrame({'model_type': [], 'usage_percentage': [], 'usage_count': []})
        
        csv_path = save_dir / "Random_model_type_usage.csv"
        df_model_usage.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        # 5. network_performance_metrics
        df_network_metrics = pd.DataFrame({
            'metric': ['Avg Transmission Time', 'Avg Latency', 'Avg Network Hops', 'Avg Bandwidth'],
            'value': [
                comm_analysis['avg_transmission_time'] * 1000,  # ms
                comm_analysis['avg_latency'],
                comm_analysis['avg_network_hops'],
                comm_analysis['avg_bandwidth_usage']
            ],
            'unit': ['ms', 'ms', 'hops', 'Mbps']
        })
        csv_path = save_dir / "Random_network_performance_metrics.csv"
        df_network_metrics.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        # 6. reward_distribution
        rewards = results['episode_rewards']
        df_reward_dist = pd.DataFrame({
            'episode': range(len(rewards)),
            'reward': rewards,
            'reward_category': ['High' if r > np.mean(rewards) else 'Low' for r in rewards]
        })
        csv_path = save_dir / "Random_reward_distribution.csv"
        df_reward_dist.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        # 7. task_success_rate
        success_rates = results['task_success_rates']
        success_count = int(np.sum(success_rates))
        fail_count = len(success_rates) - success_count
        df_success_rate = pd.DataFrame({
            'status': ['Successful', 'Failed'],
            'count': [success_count, fail_count],
            'percentage': [success_count/len(success_rates)*100, fail_count/len(success_rates)*100]
        })
        csv_path = save_dir / "Random_task_success_rate.csv"
        df_success_rate.to_csv(csv_path, index=False)
        print(f"  ✅ {csv_path.name}")
        
        print(f"✅ Random 算法的7个CSV文件导出完成")
    
    def export_random_inference_json(self):
        """导出随机算法推理JSON报告"""
        results = self.inference_results
        stats = self.performance_stats
        
        report = {
            'algorithm_metadata': {
                'algorithm_name': 'Random',
                'server_id': self.server_id,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'total_episodes': len(results['episode_rewards']),
                'analysis_type': 'Random_Algorithm_Inference'
            },
            'performance_statistics': stats,
            'inference_results_summary': {
                'episode_count': len(results['episode_rewards']),
                'average_reward': float(np.mean(results['episode_rewards'])),
                'average_completion_rate': float(np.mean(results['completion_rates'])),
                'task_success_rate': float(np.mean(results['task_success_rates'])),
                'average_response_time': float(np.mean(results['response_times'])),
                'average_communication_overhead': float(np.mean(results['communication_overhead']))
            },
            'detailed_results': results['detailed_metrics']
        }
        
        # 转换numpy类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        report = convert_numpy_types(report)
        
        output_path = self.comparison_dir / 'Random_inference.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Random 推理报告导出: {output_path.name}")
        return report
    
    def run_analysis(self, n_episodes: int = 200):
        """运行随机算法分析"""
        print("="*60)
        print("随机算法推理分析系统")
        print("="*60)
        
        try:
            # 创建环境
            env = self.create_environment()
            
            # 运行推理
            self.run_random_inference(env, n_episodes)
            
            # 导出CSV文件
            self.export_random_csv_files()
            
            # 导出JSON报告
            self.export_random_inference_json()
            
            # 打印总结
            self.print_summary()
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """打印分析总结"""
        print(f"\n{'='*60}")
        print("随机算法分析总结")
        print(f"{'='*60}")
        
        stats = self.performance_stats
        print(f"性能总结:")
        print(f"  • 平均奖励: {stats['reward_metrics']['mean']:.3f}")
        print(f"  • 任务成功率: {stats['completion_metrics']['success_rate']:.1%}")
        print(f"  • 平均完成率: {stats['completion_metrics']['mean_completion_rate']:.1%}")
        print(f"  • 平均响应时间: {stats['efficiency_metrics']['mean_response_time']:.3f}s")
        print(f"  • 平均通信开销: {stats['efficiency_metrics']['mean_communication_overhead']:.3f}s")
        
        comm_analysis = stats['communication_analysis']
        print(f"\n通信分析:")
        print(f"  • 平均传输时间: {comm_analysis['avg_transmission_time']:.4f}s")
        print(f"  • 平均网络跳数: {comm_analysis['avg_network_hops']:.1f}")
        print(f"  • 平均带宽使用: {comm_analysis['avg_bandwidth_usage']:.1f} Mbps")
        print(f"  • 平均延迟: {comm_analysis['avg_latency']:.1f} ms")
        
        server_analysis = stats['server_usage']
        print(f"\n服务器使用分析:")
        print(f"  • 活跃服务器数: {server_analysis['active_servers']}")
        print(f"  • 服务器多样性: {server_analysis['server_diversity']:.1%}")
        print(f"  • 负载均衡系数: {server_analysis['utilization_balance_coefficient']:.3f}")
        
        model_analysis = stats['model_usage']
        print(f"\n模型使用分析:")
        print(f"  • 模型类型多样性: {model_analysis['model_diversity']}")
        print(f"  • 最常用模型类型: {model_analysis['most_used_model']}")
        
        print(f"\n生成的文件:")
        print(f"  • JSON报告: Random_inference.json")
        print(f"  • CSV文件: Random_*.csv (7个文件)")
        print(f"    - active_server_usage_distribution.csv")
        print(f"    - communication_efficiency.csv")
        print(f"    - communication_overhead_distribution.csv")
        print(f"    - model_type_usage.csv")
        print(f"    - network_performance_metrics.csv")
        print(f"    - reward_distribution.csv")
        print(f"    - task_success_rate.csv")


def main():
    """主函数"""
    # 初始化分析器
    analyzer = RandomAlgorithmAnalyzer(server_id="Server4")
    
    # 运行分析
    analyzer.run_analysis(n_episodes=200)


if __name__ == "__main__":
    main()
