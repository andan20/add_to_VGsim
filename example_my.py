from final import run_optimization
from setup import setup_simulator

run_optimization(setup_simulator, 170, [60, 90],
                 [0.4, 0.8], [0.02, 0.05], 5000, 1,
                 3, 40, 10, 3, 2, 2,
                 0.025,
                 "result_out.html", "result_imp.html",
                 "result_imp.png",
                 robustness_runs=5,
                 output_robustness_html="result_robustness.html",
                 output_robustness_png="result_robustness.png")

print("конец")