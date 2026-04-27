// I/O — CSV / Parquet read+write and from_columns.
//
// Read paths normalise to a single chunk via CombineChunks: Arrow's CSV
// reader returns multiple chunks for large files, but the rest of the
// engine (filter/sort/...) assumes column->chunk(0) covers the whole array.
// Combining once at the boundary keeps the hot paths simple and O(1) for
// per-row access.
//
// The two-arg read_csv/read_parquet variants are the projection-pushdown
// hooks used by the lazy compiler — they only materialise the listed
// columns. CSV uses ConvertOptions::include_columns; Parquet uses
// ReadTable(indices) which can skip entire column chunks at the file level.

#include "../EagerDataFrame.h"
#include <arrow/csv/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <arrow/csv/writer.h>
#include <parquet/arrow/writer.h>
#include <stdexcept>

namespace dataframelib {

EagerDataFrame EagerDataFrame::read_csv(const std::string& path) {
    auto infile_result = arrow::io::ReadableFile::Open(path);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open CSV file: " + path +
                                 " | Error: " + infile_result.status().ToString());
    }
    auto infile = *infile_result;

    auto read_options = arrow::csv::ReadOptions::Defaults();
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    auto convert_options = arrow::csv::ConvertOptions::Defaults();

    auto reader_result = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(), infile,
        read_options, parse_options, convert_options);

    if (!reader_result.ok()) {
        throw std::runtime_error("Failed to create CSV reader: " + reader_result.status().ToString());
    }
    auto reader = *reader_result;

    auto table_result = reader->Read();
    if (!table_result.ok()) {
        throw std::runtime_error("Failed to read CSV data: " + table_result.status().ToString());
    }

    return EagerDataFrame((*table_result)->CombineChunks(arrow::default_memory_pool()).ValueOrDie());
}

EagerDataFrame EagerDataFrame::read_parquet(const std::string& path) {
    auto infile_result = arrow::io::ReadableFile::Open(path);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open Parquet file: " + path +
                                 " | Error: " + infile_result.status().ToString());
    }
    auto infile = *infile_result;

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    auto open_status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrow_reader);
    if (!open_status.ok()) {
        throw std::runtime_error("Failed to open Parquet reader: " + open_status.ToString());
    }

    std::shared_ptr<arrow::Table> table;
    auto read_status = arrow_reader->ReadTable(&table);
    if (!read_status.ok()) {
        throw std::runtime_error("Failed to read Parquet data: " + read_status.ToString());
    }

    return EagerDataFrame(table->CombineChunks(arrow::default_memory_pool()).ValueOrDie());
}

// Projection-pushdown read variants. Tell the underlying reader to only
// materialise the listed columns. CSV: ConvertOptions::include_columns
// avoids parsing/converting unused columns. Parquet: ReadTable(indices)
// skips entire column chunks at the file level.
EagerDataFrame EagerDataFrame::read_csv(const std::string& path,
                                        const std::vector<std::string>& columns) {
    if (columns.empty()) return read_csv(path);

    auto infile_result = arrow::io::ReadableFile::Open(path);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open CSV file: " + path +
                                 " | Error: " + infile_result.status().ToString());
    }
    auto infile = *infile_result;

    auto read_options = arrow::csv::ReadOptions::Defaults();
    auto parse_options = arrow::csv::ParseOptions::Defaults();
    auto convert_options = arrow::csv::ConvertOptions::Defaults();
    convert_options.include_columns = columns;
    convert_options.include_missing_columns = false;

    auto reader_result = arrow::csv::TableReader::Make(
        arrow::io::default_io_context(), infile,
        read_options, parse_options, convert_options);
    if (!reader_result.ok()) {
        throw std::runtime_error("Failed to create CSV reader: " + reader_result.status().ToString());
    }

    auto table_result = (*reader_result)->Read();
    if (!table_result.ok()) {
        throw std::runtime_error("Failed to read CSV data: " + table_result.status().ToString());
    }

    return EagerDataFrame((*table_result)->CombineChunks(arrow::default_memory_pool()).ValueOrDie());
}

EagerDataFrame EagerDataFrame::read_parquet(const std::string& path,
                                            const std::vector<std::string>& columns) {
    if (columns.empty()) return read_parquet(path);

    auto infile_result = arrow::io::ReadableFile::Open(path);
    if (!infile_result.ok()) {
        throw std::runtime_error("Failed to open Parquet file: " + path);
    }
    auto infile = *infile_result;

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
    auto open_status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &arrow_reader);
    if (!open_status.ok()) {
        throw std::runtime_error("Failed to open Parquet reader: " + open_status.ToString());
    }

    // Map column names → indices in the file's schema, dropping anything we
    // can't find (the user may have requested a derived column that's added
    // upstream by with_column).
    std::shared_ptr<arrow::Schema> schema;
    auto schema_status = arrow_reader->GetSchema(&schema);
    if (!schema_status.ok()) {
        throw std::runtime_error("Failed to read Parquet schema: " + schema_status.ToString());
    }

    std::vector<int> indices;
    indices.reserve(columns.size());
    for (const auto& name : columns) {
        int idx = schema->GetFieldIndex(name);
        if (idx >= 0) indices.push_back(idx);
    }
    if (indices.empty()) return read_parquet(path);  // fall back if nothing matched

    std::shared_ptr<arrow::Table> table;
    auto read_status = arrow_reader->ReadTable(indices, &table);
    if (!read_status.ok()) {
        throw std::runtime_error("Failed to read Parquet data: " + read_status.ToString());
    }

    return EagerDataFrame(table->CombineChunks(arrow::default_memory_pool()).ValueOrDie());
}

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

    auto status = parquet::arrow::WriteTable(*table_, arrow::default_memory_pool(), outfile, 1024);
    if (!status.ok()) {
        throw std::runtime_error("Failed to write Parquet: " + status.ToString());
    }
}

EagerDataFrame EagerDataFrame::from_columns(
    const std::vector<std::pair<std::string, std::shared_ptr<arrow::Array>>>& cols) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    for (const auto& [name, array] : cols) {
        fields.push_back(arrow::field(name, array->type()));
        arrays.push_back(array);
    }

    auto schema = std::make_shared<arrow::Schema>(fields);
    auto table = arrow::Table::Make(schema, arrays);
    return EagerDataFrame(table->CombineChunks(arrow::default_memory_pool()).ValueOrDie());
}

}
