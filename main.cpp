#include "DataFrameLib/Eager.hpp"
#include "DataFrameLib/Expr.hpp"
#include "DataFrameLib/Lazy.hpp"
#include <iostream>
#include <fstream>

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
        auto df = EagerDataFrame::read_csv("test_data.csv");
        df.print();

        std::cout << "\n=== 2. Testing select() ===" << std::endl;
        auto df_selected = df.select({"name", "age", "salary"});
        df_selected.print();

        std::cout << "\n=== 3. Testing filter(age > 30) ===" << std::endl;
        auto df_filtered = df.filter(col("age") > 30); 
        df_filtered.print();

        std::cout << "\n=== 4. Testing with_column() ===" << std::endl;
        // Adding a column that doubles the salary
        auto df_mutated = df.with_column("double_salary", col("salary") + col("salary"));
        df_mutated.print();

        std::cout << "\n=== 5. Testing sort(salary, desc) ===" << std::endl;
        // Sort by salary descending (false means not ascending)
        auto df_sorted = df.sort({"salary"}, false);
        df_sorted.print();

        std::cout << "\n=== 6. Testing head(2) ===" << std::endl;
        auto df_head = df_sorted.head(2);
        df_head.print();

        std::cout << "\n=== 7. Testing Lazy DAG Construction & Graphviz ===" << std::endl;
        // Notice we are using the lazy scan, not eager read!
        auto lazy_df = LazyDataFrame::scan_csv("test_data.csv");
        
        // Build a complex chain of operations
        auto lazy_plan = lazy_df
            .filter(col("age") > 30)
            .select({"name", "salary"})
            .filter(col("salary") > lit(90000.0f));

        // This should output a 'plan.png' file in your build directory
        lazy_plan.explain("plan.png");

        std::cout << "\n=== 8. Testing INNER JOIN ===" << std::endl;
        
        // 1. Our left table is the 'df' we loaded earlier (has 'dept' column)
        
        // 2. Let's create a quick right table using EagerDataFrame's CSV reader manually
        std::ofstream dept_file("dept_data.csv");
        dept_file << "dept,manager\n";
        dept_file << "Engineering,Grace Hopper\n";
        dept_file << "Sales,Jordan Belfort\n";
        dept_file.close();
        
        auto dept_df = EagerDataFrame::read_csv("dept_data.csv");
        std::cout << "Right Table (Departments):" << std::endl;
        dept_df.print();

        // 3. Perform the Join!
        auto joined_df = df.join(dept_df, "dept", "inner");
        std::cout << "Joined Table:" << std::endl;
        joined_df.print();

        std::cout << "All basic Eager operations passed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}