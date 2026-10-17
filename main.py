from config import ensure_directories, apply_statistical_style, DATA_URL
from data_prep import load_and_clean_data, check_assumptions
from hypothesis_tests import run_all_hypothesis_tests
from visualizations import generate_all_plots

def main():
    print("=" * 60)
    print("STARTING WEEK 3: STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
    print("=" * 60)

    ensure_directories()
    apply_statistical_style()

    df_clean = load_and_clean_data(DATA_URL)
    
    check_assumptions(df_clean)
    run_all_hypothesis_tests(df_clean)
    generate_all_plots(df_clean)

    print("\n" + "=" * 60)
    print("WEEK 3 PIPELINE COMPLETED SUCCESSFULLY!")
    print("Check the 'outputs/week3_plots/' folder for your deliverables.")
    print("=" * 60)

if __name__ == "__main__":
    main()