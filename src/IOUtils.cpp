#include "DataFrameLib/Eager.hpp"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/csv/writer.h>
#include <parquet/arrow/writer.h>
#include <stdexcept>

EagerDataFrame EagerDataFrame::read_csv(const std::string& path) {
    // Opening the file stream
    auto infile_result = arrow::io::ReadableFile::Open(path);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open CSV file: " + path + 
                                 " | Error: " + infile_result.status().ToString());
    }
    auto infile = *infile_result;

    // Setting up Arrow's default CSV reading options
    auto read_options = arrow::csv::ReadOptions::Defaults();
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    auto convert_options = arrow::csv::ConvertOptions::Defaults();

    // Initializing the TableReader
    auto reader_result = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(),
        infile,
        read_options,
        parse_options,
        convert_options
    );

    if (!reader_result.ok()) {
        throw std::runtime_error("Failed to create CSV reader: " + 
                                 reader_result.status().ToString());
    }
    auto reader = *reader_result;

    // Actually reading the data into an arrow::Table
    auto table_result = reader->Read();
    if (!table_result.ok()) {
        throw std::runtime_error("Failed to read CSV data: " + 
                                 table_result.status().ToString());
    }

    // Wrapping it in our EagerDataFrame and returning
    return EagerDataFrame(*table_result);
}


EagerDataFrame EagerDataFrame::read_parquet(const std::string& path) {
    // Opening the file stream (same as CSV)
    auto infile_result = arrow::io::ReadableFile::Open(path);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open Parquet file: " + path + 
                                 " | Error: " + infile_result.status().ToString());
    }
    auto infile = *infile_result;

    // Initializing the Parquet FileReader
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    auto open_status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrow_reader);
    
    if (!open_status.ok()) {
        throw std::runtime_error("Failed to open Parquet reader: " + open_status.ToString());
    }

    // Reading the data into an arrow::Table
    std::shared_ptr<arrow::Table> table;
    auto read_status = arrow_reader->ReadTable(&table);
    
    if (!read_status.ok()) {
        throw std::runtime_error("Failed to read Parquet data: " + read_status.ToString());
    }

    // Wrapping and returning
    return EagerDataFrame(table);
}

// Writing functions for CSV and Parquet formats
void EagerDataFrame::write_csv(const std::string& path) const {
    auto outfile_result = arrow::io::FileOutputStream::Open(path);
    if (!outfile_result.ok()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    auto outfile = *outfile_result;

    auto write_options = arrow::csv::WriteOptions::Defaults();
    auto status = arrow::csv::WriteCSV(*table_, write_options, outfile.get());
    
    if (!status.ok()) {
        throw std::runtime_error("Failed to write CSV: " + status.ToString());
    }
}

void EagerDataFrame::write_parquet(const std::string& path) const {
    auto outfile_result = arrow::io::FileOutputStream::Open(path);
    if (!outfile_result.ok()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    auto outfile = *outfile_result;

    // Write table with default Parquet properties
    auto status = parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(), outfile, 1024);
    
    if (!status.ok()) {
        throw std::runtime_error("Failed to write Parquet: " + status.ToString());
    }
}

//Create a dataframe from a map of column name to arrow array
EagerDataFrame EagerDataFrame::from_columns(const std::unordered_map<std::string, std::shared_ptr<arrow::Array>>& col_map) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    
    for (const auto& [name, array] : col_map) {
        fields.push_back(arrow::field(name, array->type()));
        arrays.push_back(array);
    }
    
    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table = arrow::Table::Make(schema, arrays);
    
    return EagerDataFrame(table);
}