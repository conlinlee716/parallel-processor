"""完整的Nsight Profiling wrapper - Windows完全兼容版"""
import subprocess
import os
import sys
import platform
import json
import sqlite3
from pathlib import Path
import pandas as pd


class NSightProfiler:
    """Nsight Systems profiling with comprehensive analysis"""
    
    def __init__(self, output_dir="profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.is_windows = platform.system() == "Windows"
        print(f"[Init] Output directory: {self.output_dir.absolute()}")
        print(f"[Init] Platform: {platform.system()}\n")
    
    @staticmethod
    def check_nsys_installed():
        """检查nsys是否安装"""
        try:
            result = subprocess.run(["nsys", "--version"], capture_output=True, text=True)
            print(f"[Check] nsys version: {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            print("[ERROR] nsys not found!")
            return False
    
    def profile_systems(self, script_name, output_file=None):
        """
        使用Nsight Systems进行profiling
        
        关键参数：
        - --sample=on: 启用采样
        - --sampling-period=1: 采样周期（毫秒）
        - --cudabacktrace=full: 完整CUDA回溯
        """
        if output_file is None:
            output_file = self.output_dir / "profile.nsys-rep"
        else:
            output_file = self.output_dir / output_file
        
        output_file = str(output_file)
        
        print(f"\n{'='*70}")
        print(f"[Profiling] Starting Nsight Systems profiling...")
        print(f"{'='*70}")
        print(f"Target script: {script_name}")
        print(f"Output file: {output_file}\n")
        
        # Windows兼容的命令
        cmd = [
            "nsys",
            "profile",
            "--stats=true",                    # 生成统计
            "--force-overwrite=true",          # 覆盖现有
            "--trace=cuda,nvtx",               # Windows支持的参数
            "--sample=on",                     # 启用采样
            "--cpuctxsw=on",                   # CPU上下文切换
            f"--output={output_file}",
            "python",
            script_name
        ]
        
        print(f"[Command] {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                
                if file_size < 1024 * 100:  # 小于100KB
                    print(f"⚠️  [Warning] Profile file very small ({file_size / 1024:.1f} KB)")
                    print(f"    This may indicate the script ran too quickly.")
                    print(f"    For better profiling, try:")
                    print(f"      • Increase loop iterations")
                    print(f"      • Use larger input data")
                    print(f"      • Add warmup runs\n")
                else:
                    print(f"✓ [Success] Profile file created")
                    print(f"  Path: {os.path.abspath(output_file)}")
                    print(f"  Size: {file_size / 1024 / 1024:.1f} MB\n")
                
                return output_file
            else:
                print(f"✗ [Failed] Profile file was not created")
                print(f"\nStderr:\n{result.stderr}")
                return None
                
        except Exception as e:
            print(f"✗ [Error] {e}")
            return None
    
    def convert_to_sqlite_and_analyze(self, nsys_file):
        """
        将.nsys-rep转换为SQLite并进行分析
        Windows上推荐用SQLite而不是HTML
        """
        db_file = f"{nsys_file}.db"
        
        print(f"\n{'='*70}")
        print(f"[SQLite Export] Converting to database...")
        print(f"{'='*70}\n")
        
        cmd = [
            "nsys",
            "export",
            "--type", "sqlite",
            f"--output", db_file,
            nsys_file
        ]
        
        print(f"[Command] {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if os.path.exists(db_file):
                file_size = os.path.getsize(db_file)
                print(f"✓ [Success] SQLite database created")
                print(f"  Path: {os.path.abspath(db_file)}")
                print(f"  Size: {file_size / 1024 / 1024:.1f} MB\n")
                
                return db_file
            else:
                print(f"✗ [Failed] SQLite export failed")
                print(f"Error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"✗ [Error] {e}")
            return None
    
    def analyze_sqlite_database(self, db_file):
        """
        从SQLite数据库中提取关键性能指标
        """
        print(f"\n{'='*70}")
        print(f"[Database Analysis] Analyzing performance data...")
        print(f"{'='*70}\n")
        
        try:
            conn = sqlite3.connect(db_file)
            
            # ===== 查询1：CUDA Kernel统计 =====
            print("【CUDA Kernel Execution Summary】")
            print("-" * 70)
            
            kernel_query = """
            SELECT 
                COUNT(*) as total_kernels,
                ROUND(AVG((end - start)/1e3), 2) as avg_duration_us,
                ROUND(MAX((end - start)/1e3), 2) as max_duration_us,
                ROUND(MIN((end - start)/1e3), 2) as min_duration_us,
                ROUND(SUM((end - start)/1e3), 2) as total_duration_us
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
            
            try:
                kernel_stats = pd.read_sql_query(kernel_query, conn)
                if not kernel_stats.empty:
                    print(kernel_stats.to_string(index=False))
                    print()
            except Exception as e:
                print(f"(Kernel table not available: {e})\n")
            
            # ===== 查询2：Top Kernels by Duration =====
            print("\n【Top 10 Hotspot Kernels】")
            print("-" * 70)
            
            top_kernels = """
            SELECT 
                s.value as kernel_name,
                COUNT(*) as call_count,
                ROUND(SUM((end - start)/1e3), 2) as total_us,
                ROUND(AVG((end - start)/1e3), 2) as avg_us,
                ROUND(100.0 * SUM((end - start)/1e3) / 
                    (SELECT SUM((end - start)/1e3) FROM CUPTI_ACTIVITY_KIND_KERNEL), 2) as percent        
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            GROUP BY kernel_name
            ORDER BY total_us DESC
            LIMIT 10
            """
            
            try:
                top_df = pd.read_sql_query(top_kernels, conn)
                if not top_df.empty:
                    print(top_df.to_string(index=False))
                else:
                    print("(No kernel data available)")
                print()
            except Exception as e:
                print(f"(Kernel analysis failed: {e})\n")
            
            # ===== 查询3：内存传输统计 =====
            print("\n【GPU Memory Transfer Statistics】")
            print("-" * 70)
            
            memcpy_query = """
            SELECT 
                COUNT(*) as transfer_count,
                ROUND(SUM(bytes) / 1e9, 3) as total_gb,
                ROUND(AVG(bytes) / 1e6, 3) as avg_mb,
                ROUND(AVG((end - start)/1e3), 2) as avg_latency_us,
                ROUND(SUM(bytes) * 1e6 / SUM((end - start)/1e3) / 1e9, 2) as bandwidth_gbps
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
            
            try:
                memcpy_stats = pd.read_sql_query(memcpy_query, conn)
                if not memcpy_stats.empty:
                    print(memcpy_stats.to_string(index=False))
                else:
                    print("(No memory transfer data)")
                print()
            except Exception as e:
                print(f"(Memory analysis failed: {e})\n")
            
            # ===== 查询4：按类型分类的内存传输 =====
            print("\n【Memory Transfer by Type】")
            print("-" * 70)
            
            memcpy_type_query = """
            SELECT 
                CASE 
                    WHEN copyKind = 1 THEN 'H2D (Host→Device)'
                    WHEN copyKind = 2 THEN 'D2H (Device→Host)'
                    WHEN copyKind = 3 THEN 'D2D (Device→Device)'
                    ELSE 'Unknown'
                END as transfer_type,
                COUNT(*) as count,
                ROUND(SUM(bytes) / 1e9, 3) as total_gb,
                ROUND(SUM(bytes) * 1e6 / SUM((end - start)/1e3) / 1e9, 2) as bandwidth_gbps
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            GROUP BY copyKind
            """
            
            try:
                memcpy_type = pd.read_sql_query(memcpy_type_query, conn)
                if not memcpy_type.empty:
                    print(memcpy_type.to_string(index=False))
                else:
                    print("(No memory transfer data)")
                print()
            except Exception as e:
                print(f"(Transfer type analysis failed: {e})\n")
            
            # ===== 查询5：GPU活动时间线 =====
            print("\n【Execution Timeline】")
            print("-" * 70)
            
            timeline_query = """
            SELECT 
                COUNT(*) as total_events,
                ROUND(MIN(start) / 1e6, 3) as first_event_sec,
                ROUND(MAX(end) / 1e6, 3) as last_event_sec,
                ROUND((MAX(end) - MIN(start)) / 1e6, 3) as total_duration_sec
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
            
            try:
                timeline = pd.read_sql_query(timeline_query, conn)
                if not timeline.empty:
                    print(timeline.to_string(index=False))
                else:
                    print("(No timeline data)")
                print()
            except Exception as e:
                print(f"(Timeline analysis failed: {e})\n")
            
            conn.close()
            
            print(f"\n✓ [Success] Analysis complete\n")
            
            return True
            
        except Exception as e:
            print(f"✗ [Error] {e}\n")
            return False
    
    def generate_html_from_database(self, db_file, nsys_file):
        """
        生成HTML报告（Windows版本 - 从SQLite数据手动生成）
        """
        html_file = f"{nsys_file}.html"
        
        print(f"\n{'='*70}")
        print(f"[HTML Generation] Creating HTML report from database...")
        print(f"{'='*70}\n")
        
        try:
            conn = sqlite3.connect(db_file)
            
            # 收集数据
            kernels = pd.read_sql_query(
                """SELECT s.value as demangledName, COUNT(*) as calls, SUM((end - start)/1e3) as total_us 
                   FROM CUPTI_ACTIVITY_KIND_KERNEL k
                   JOIN StringIds s ON k.demangledName = s.id
                   GROUP BY demangledName 
                   ORDER BY total_us DESC LIMIT 20""",
                conn
            )
            
            conn.close()
            
            # 准备图表数据
            kernel_names = kernels['demangledName'].head(10).tolist()
            kernel_durations = kernels['total_us'].head(10).tolist()
            
            # 使用json.dumps确保字符串正确转义，并在内部添加省略号
            labels = []
            for name in kernel_names:
                if len(name) > 30:
                    truncated_name = name[:30] + '...'
                else:
                    truncated_name = name
                labels.append(truncated_name)
            
            # 确保数据不是空的
            if not labels:
                labels = ['No data available']
                kernel_durations = [0]
            
            # 生成HTML
            # 生成JSON数据
            import json
            json_labels_str = json.dumps(labels, ensure_ascii=False)
            json_data_str = json.dumps(kernel_durations)
            
            # 构建HTML内容
            html_content = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>GPU Profiling Report</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: #333;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        border-radius: 10px;
                        padding: 30px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    }}
                    h1 {{
                        color: #667eea;
                        border-bottom: 3px solid #667eea;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #764ba2;
                        margin-top: 30px;
                    }}
                    .chart-container {{
                        position: relative;
                        height: 400px;
                        margin: 20px 0;
                        background: #f9f9f9;
                        border-radius: 8px;
                        padding: 20px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th, td {{
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #667eea;
                        color: white;
                    }}
                    tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    .info-box {{
                        background: #e3f2fd;
                        border-left: 4px solid #2196F3;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 4px;
                    }}
                    .warning-box {{
                        background: #fff3e0;
                        border-left: 4px solid #ff9800;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 4px;
                    }}
                    footer {{
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        color: #666;
                        font-size: 12px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🚀 GPU Profiling Report</h1>
                    
                    <div class="info-box">
                        <strong>ℹ️ Report Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                        <strong>Profile File:</strong> {nsys_file}<br>
                        <strong>Database:</strong> {db_file}
                    </div>
                    
                    <h2>Top CUDA Kernels</h2>
            """
            
            # 添加表格
            html_content += """
                    <table>
                        <tr>
                            <th>Kernel Name</th>
                            <th>Call Count</th>
                            <th>Total Duration (μs)</th>
                            <th>% of Total</th>
                        </tr>
            """
            
            if not kernels.empty:
                total_duration = kernels['total_us'].sum()
                for _, row in kernels.iterrows():
                    percent = (row['total_us'] / total_duration * 100) if total_duration > 0 else 0
                    html_content += f"""
                        <tr>
                            <td>{row['demangledName']}</td>
                            <td>{int(row['calls'])}</td>
                            <td>{row['total_us']:.2f}</td>
                            <td>{percent:.1f}%</td>
                        </tr>
                    """
            
            html_content += """
                    </table>
                    
                    <h2>📊 Performance Chart</h2>
                    <div class="chart-container">
                        <canvas id="kernelChart"></canvas>
                    </div>
                    
                    <h2>💡 Recommendations</h2>
                    <ul>
                        <li>Check hotspot kernels for optimization opportunities</li>
                        <li>Profile with larger datasets for more meaningful results</li>
                        <li>Use Nsight Compute for kernel-level analysis</li>
                        <li>Monitor memory bandwidth utilization</li>
                    </ul>
                    
                    <div class="warning-box">
                        <strong>📝 Note:</strong> For detailed timeline visualization, use:<br>
                        <code>nsys-ui {nsys_file}</code>
                    </div>
                    
                    <footer>
                        Generated by GPU Profiler | NVIDIA Nsight Systems {subprocess.run(['nsys', '--version'], capture_output=True, text=True).stdout.strip()}
                    </footer>
                </div>
                
                <script>
                    const ctx = document.getElementById('kernelChart').getContext('2d');
                    const labels = {LABELS_PLACEHOLDER};
                    const data = {DATA_PLACEHOLDER};
                    
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Duration (μs)',
                                data: data,
                                backgroundColor: '#667eea',
                                borderColor: '#764ba2',
                                borderWidth: 2
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                </script>
            </body>
            </html>
            """
            
            # 替换占位符为实际数据
            html_content = html_content.replace('{LABELS_PLACEHOLDER}', json_labels_str)
            html_content = html_content.replace('{DATA_PLACEHOLDER}', json_data_str)
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size = os.path.getsize(html_file)
            print(f"✓ [Success] HTML report created")
            print(f"  Path: {os.path.abspath(html_file)}")
            print(f"  Size: {file_size / 1024:.1f} KB\n")
            
            return html_file
            
        except Exception as e:
            print(f"✗ [Error] HTML generation failed: {e}\n")
            return None
    
    def run_full_analysis(self, script_name):
        """完整分析流程"""
        print(f"\n{'#'*70}")
        print(f"# GPU PROFILING - COMPLETE WORKFLOW")
        print(f"{'#'*70}")
        
        if not self.check_nsys_installed():
            return None
        
        # 步骤1：Profiling
        nsys_file = self.profile_systems(script_name)
        if not nsys_file:
            print("✗ Profiling failed")
            return None
        
        results = {'profile_file': nsys_file}
        
        # 步骤2：转换为SQLite
        db_file = self.convert_to_sqlite_and_analyze(nsys_file)
        if db_file:
            results['db_file'] = db_file
            
            # 步骤3：分析数据库
            self.analyze_sqlite_database(db_file)
            
            # 步骤4：生成HTML
            html_file = self.generate_html_from_database(db_file, nsys_file)
            if html_file:
                results['html_file'] = html_file
        
        # 步骤5：总结
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """打印最终总结"""
        print(f"\n{'='*70}")
        print(f"PROFILING COMPLETE - SUMMARY")
        print(f"{'='*70}\n")
        
        print("📁 Generated Files:")
        for key, path in results.items():
            if path:
                print(f"  • {key:15} : {os.path.abspath(path)}")
        
        print(f"\n{'='*70}")
        print(f"🔍 NEXT STEPS:")
        print(f"{'='*70}\n")
        
        if 'html_file' in results and results['html_file']:
            html = os.path.abspath(results['html_file'])
            print(f"1️⃣  Open HTML Report:\n")
            print(f"    • Double-click: {html}")
            print(f"    • Or: start {html}\n")
        
        if 'db_file' in results and results['db_file']:
            db = os.path.abspath(results['db_file'])
            print(f"2️⃣  Query Database:\n")
            print(f"    sqlite3 {db}\n")
        
        if 'profile_file' in results:
            profile = os.path.abspath(results['profile_file'])
            print(f"3️⃣  Deep Analysis with Nsight GUI:\n")
            print(f"    nsys-ui {profile}\n")


if __name__ == "__main__":
    profiler = NSightProfiler(output_dir="profiling_results")
    results = profiler.run_full_analysis("test_end2end.py")
    
    if results:
        print("\n✅ All profiling steps completed successfully!")
    else:
        print("\n❌ Profiling failed!")
        sys.exit(1)
