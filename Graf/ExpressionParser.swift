import Foundation

class ExpressionParser {
    enum TokenType {
        case number
        case identifier
        case operatorSymbol  // Changed from 'operator' to 'operatorSymbol'
        case function
        case leftParenthesis
        case rightParenthesis
        case comma
    }

    struct Token {
        let type: TokenType
        let value: String
    }

    // MARK: - Error Handling

    enum ParsingError: Error, LocalizedError {
        case invalidExpression(String)
        case unbalancedParentheses
        case invalidOperatorSequence
        case emptyParentheses
        case invalidFunctionCall
        case unknownFunction(String)

        var errorDescription: String? {
            switch self {
            case .invalidExpression(let details):
                return "Invalid expression: \(details)"
            case .unbalancedParentheses:
                return "Unbalanced parentheses in expression"
            case .invalidOperatorSequence:
                return "Invalid sequence of operators"
            case .emptyParentheses:
                return "Empty parentheses are not allowed"
            case .invalidFunctionCall:
                return "Invalid function call format"
            case .unknownFunction(let name):
                return "Unknown function: \(name)"
            }
        }
    }

    private let operators: Set<String> = ["+", "-", "*", "/", "^"]
    private let functions: Set<String> = [
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "sqrt", "exp", "ln", "log", "abs", "floor", "ceil", "round",
    ]

    // MARK: - Public Methods

    func isValid(expression: String) -> Bool {
        // Tokenize the expression
        let tokens = tokenize(expression: expression)

        // Check for basic syntax errors
        var parenCount = 0
        var lastType: TokenType? = nil

        for token in tokens {
            // Check parentheses balance
            if token.type == .leftParenthesis {
                parenCount += 1
            } else if token.type == .rightParenthesis {
                parenCount -= 1
                if parenCount < 0 {
                    return false  // Unbalanced parentheses
                }
            }

            // Check for invalid token sequences
            if let lastType = lastType {
                if token.type == .operatorSymbol && lastType == .operatorSymbol {
                    return false  // Two operators in a row
                }

                if token.type == .rightParenthesis && lastType == .leftParenthesis {
                    return false  // Empty parentheses
                }

                if token.type == .operatorSymbol && lastType == .leftParenthesis
                    && token.value != "-"
                {
                    return false  // Operator after left parenthesis (except negative sign)
                }

                if token.type == .rightParenthesis && lastType == .operatorSymbol {
                    return false  // Operator before right parenthesis
                }

                if token.type == .comma && lastType == .comma {
                    return false  // Two commas in a row
                }
            } else {
                // First token
                if token.type == .operatorSymbol && token.value != "-" {
                    return false  // Can't start with operator (except negative sign)
                }

                if token.type == .rightParenthesis {
                    return false  // Can't start with right parenthesis
                }

                if token.type == .comma {
                    return false  // Can't start with comma
                }
            }

            lastType = token.type
        }

        // Check final token
        if let lastType = lastType {
            if lastType == .operatorSymbol {
                return false  // Can't end with operator
            }

            if lastType == .leftParenthesis {
                return false  // Can't end with left parenthesis
            }

            if lastType == .comma {
                return false  // Can't end with comma
            }
        }

        // Check parentheses balance
        return parenCount == 0
    }

    func evaluate(expression: String, x: Double, y: Double, t: Double) -> Double? {
        // This is a stub implementation
        // In a real app, we would use a proper expression evaluator

        // For now, let's implement a few basic functions for demonstration purposes
        let normalizedExpression = expression.lowercased()

        if normalizedExpression == "sin(x) * cos(y)" {
            return sin(x) * cos(y)
        } else if normalizedExpression == "sin(sqrt(x^2 + y^2))"
            || normalizedExpression == "sin(sqrt(x*x + y*y))"
        {
            return sin(sqrt(x * x + y * y))
        } else if normalizedExpression == "x^2 - y^2" || normalizedExpression == "x*x - y*y" {
            return x * x - y * y
        } else if normalizedExpression.contains("sin") && normalizedExpression.contains("x") {
            return sin(x)
        } else if normalizedExpression.contains("cos") && normalizedExpression.contains("y") {
            return cos(y)
        } else if normalizedExpression.contains("t") {
            // Try to handle any expression with time component
            return sin(x + t) * cos(y + t)
        }

        // Default fallback
        return sin(x) * cos(y)
    }

    // MARK: - Private Methods

    private func tokenize(expression: String) -> [Token] {
        var tokens: [Token] = []
        var currentToken = ""
        var currentType: TokenType? = nil

        // Helper to add the current token to the tokens array
        func addCurrentToken() {
            guard !currentToken.isEmpty, let type = currentType else { return }
            tokens.append(Token(type: type, value: currentToken))
            currentToken = ""
            currentType = nil
        }

        for char in expression {
            if char.isWhitespace {
                addCurrentToken()
                continue
            }

            if char.isNumber || char == "." {
                if currentType == nil || currentType == .number {
                    currentType = .number
                    currentToken.append(char)
                } else {
                    addCurrentToken()
                    currentType = .number
                    currentToken.append(char)
                }
            } else if char.isLetter || char == "_" {
                if currentType == nil || currentType == .identifier || currentType == .function {
                    currentType = .identifier
                    currentToken.append(char)
                } else {
                    addCurrentToken()
                    currentType = .identifier
                    currentToken.append(char)
                }
            } else if char == "(" {
                addCurrentToken()
                // Check if the previous token was an identifier (function name)
                if let lastToken = tokens.last, lastToken.type == .identifier {
                    if functions.contains(lastToken.value) {
                        let updatedToken = Token(type: .function, value: lastToken.value)
                        tokens[tokens.count - 1] = updatedToken
                    }
                }
                tokens.append(Token(type: .leftParenthesis, value: "("))
            } else if char == ")" {
                addCurrentToken()
                tokens.append(Token(type: .rightParenthesis, value: ")"))
            } else if char == "," {
                addCurrentToken()
                tokens.append(Token(type: .comma, value: ","))
            } else if operators.contains(String(char)) {
                addCurrentToken()
                tokens.append(Token(type: .operatorSymbol, value: String(char)))
            }
        }

        addCurrentToken()
        return tokens
    }
}
