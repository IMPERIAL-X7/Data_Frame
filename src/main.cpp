#include <dataframelib/dataframelib.h>
#include <iostream>
#include <fstream>

using namespace dataframelib;

void create_dummy_csv() {
    std::ofstream file("test_data.csv");
    file << "id,name,age,salary,dept\n";
    file << "1,Alice,28,85000.0,Engineering\n";
    file << "2,Bob,35,92000.0,Sales\n";
    file << "3,Charlie,22,60000.0,Engineering\n";
    file << "4,Diana,31,105000.0,Marketing\n";
    file << "5,Eve,29,95000.0,Engineering\n";
    file.close();
}

int main() {
    try {
        std::cout << "Creating dummy CSV data..." << std::endl;
        create_dummy_csv();

        std::cout << "\n=== 1. Reading CSV ===" << std::endl;
        auto df = read_csv("test_data.csv");
        df.print();

        std::cout << "\n=== 2. Testing select() ===" << std::endl;
        auto df_selected = df.select({"name", "age", "salary"});
        df_selected.print();

        std::cout << "\n=== 3. Testing filter(age > 30) ===" << std::endl;
        auto df_filtered = df.filter(col("age") > 30);
        df_filtered.print();

        std::cout << "\n=== 4. Testing with_column() ===" << std::endl;
        auto df_mutated = df.with_column("double_salary", col("salary") + col("salary"));
        df_mutated.print();

        std::cout << "\n=== 5. Testing sort(salary, desc) ===" << std::endl;
        auto df_sorted = df.sort({"salary"}, false);
        df_sorted.print();

        std::cout << "\n=== 6. Testing head(2) ===" << std::endl;
        auto df_head = df_sorted.head(2);
        df_head.print();

        std::cout << "\n=== 7. Testing Lazy DAG Construction & Graphviz ===" << std::endl;
        auto lazy_df = scan_csv("test_data.csv");
        auto lazy_plan = lazy_df
            .filter(col("age") > 30)
            .select({"name", "salary"})
            .filter(col("salary") > lit(90000.0f));
        lazy_plan.explain("plan.png");

        std::cout << "\n=== 8. Testing INNER JOIN ===" << std::endl;
        std::ofstream dept_file("dept_data.csv");
        dept_file << "dept,manager\n";
        dept_file << "Engineering,Grace Hopper\n";
        dept_file << "Sales,Jordan Belfort\n";
        dept_file.close();

        auto dept_df = read_csv("dept_data.csv");
        std::cout << "Right Table (Departments):" << std::endl;
        dept_df.print();

        auto joined_df = df.join(dept_df, {"dept"}, "inner");
        std::cout << "Joined Table:" << std::endl;
        joined_df.print();

        std::cout << "\n=== 9. Testing GroupBy & Aggregate ===" << std::endl;
        auto grouped_df = df.group_by({"dept"})
                            .aggregate({{"salary", "mean"}, {"id", "count"}});
        // (^ vector<pair<string,string>> — new aggregate signature)
        std::cout << "Aggregated Table:" << std::endl;
        grouped_df.print();

        std::cout << "\n=== 10. Testing File Output ===" << std::endl;
        df.write_csv("output.csv");
        df.write_parquet("output.parquet");
        std::cout << "Successfully wrote output.csv and output.parquet!" << std::endl;

        std::cout << "\n=== 11. Testing Lazy Execution & Optimization ===" << std::endl;
        auto lazy_df_opt = scan_csv("test_data.csv");
        auto lazy_plan_opt = lazy_df_opt
            .select({"name", "salary"})
            .filter(col("salary") > lit(90000.0f));

        lazy_plan_opt.explain("unoptimized.png");
        auto final_df = lazy_plan_opt.collect();
        final_df.print();

        std::cout << "All basic Eager operations passed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
