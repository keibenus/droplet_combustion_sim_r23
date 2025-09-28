# check_pstat.py (シェルリダイレクト版)
import pstats
import sys
import os

profile_file = 'profile_output_r10_r1_6.prof'
# output_file = 'profile_analysis.txt' # ファイル名はコマンドラインで指定
stats_to_show = 50

try:
    if not os.path.exists(profile_file):
        print(f"Error: Profile file '{profile_file}' not found.", file=sys.stderr) # エラーは標準エラー出力へ
        sys.exit(1)
    if os.path.getsize(profile_file) == 0:
        print(f"Error: Profile file '{profile_file}' is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading profile data from: {profile_file}", file=sys.stderr) # 情報メッセージも標準エラー出力へ
    p = pstats.Stats(profile_file)
    print("Profile data loaded successfully.", file=sys.stderr)

    if p.total_calls == 0:
         print("Error: No function calls recorded in the profile data.", file=sys.stderr)
         sys.exit(1)

    # --- 結果を標準出力へ直接 print ---
    print(f"Profiling analysis for: {profile_file}\n")
    print(f"Total function calls: {p.total_calls}")
    print(f"Total time: {p.total_tt:.4f} seconds\n")

    print("=" * 30 + f" Sorted by CUMULATIVE time (Top {stats_to_show}) " + "=" * 30)
    p.strip_dirs()
    p.sort_stats('cumulative')
    p.print_stats(stats_to_show) # 標準出力へ

    print("\n" + "=" * 30 + f" Sorted by TOTAL time (tottime) (Top {stats_to_show}) " + "=" * 30)
    p.sort_stats('tottime')
    p.print_stats(stats_to_show) # 標準出力へ
    # --- print ここまで ---

    print(f"\nAnalysis complete.", file=sys.stderr)

except Exception as e:
    print(f"An error occurred during profile analysis: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)