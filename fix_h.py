with open("src/ExpressionSystem.h", "r") as f:
    text = f.read()

text = text.replace("Expr lit(int v);\nExpr lit(double v);\nExpr lit(float v);\nExpr lit(const char* v);",
"Expr lit(int v);\nExpr lit(int64_t v);\nExpr lit(double v);\nExpr lit(float v);\nExpr lit(const char* v);")

with open("src/ExpressionSystem.h", "w") as f:
    f.write(text)

with open("src/ExpressionSystem.cpp", "r") as f:
    text = f.read()

import re
text = re.sub(r'Expr lit\(int v\) \{ return Expr\(LitValue\(v\)\); \}.*?Expr lit\(const char\* v\) \{ return Expr\(LitValue\(std::string\(v\)\)\); \}',
"""Expr lit(int v) { return Expr(LitValue(v)); }
Expr lit(int64_t v) { return Expr(LitValue(static_cast<int>(v))); }
Expr lit(double v) { return Expr(LitValue(v)); }
Expr lit(float v) { return Expr(LitValue(static_cast<double>(v))); }
Expr lit(const char* v) { return Expr(LitValue(std::string(v))); }""", text, flags=re.DOTALL)

with open("src/ExpressionSystem.cpp", "w") as f:
    f.write(text)
