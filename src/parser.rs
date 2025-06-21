use core::ops::Range;
mod dice_roll;
use dice_roll::roll;

type Number = f64;

#[derive(Debug)]
pub(crate) enum ErrorType {
    IllegalToken,
    UnmatchedParen,
    ExpectedValueGotParen,
    ExpectedValueGotBinaryOperator,
    ExpectedNonValue,
    #[allow(unused)]
    BadSemantics,
    Internal,
}

#[derive(Debug)]
pub(crate) struct Error {
    pub(crate) error_type: ErrorType,
    pub(crate) position: usize,
}

pub(crate) fn do_dice_command(command: &str) -> Result<Number, Error> {
    let tokens = lex(command);
    let ast = parse(command, tokens)?;
    #[cfg(test)]
    println!("{ast:?}");
    Ok(ast.execute()?.as_number())
}

struct Lex<'a> {
    str: &'a str,
    start_offset: usize,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Token {
    Empty(usize),
    Whitespace(Range<usize>),
    OpenParen(Range<usize>),
    CloseParen(Range<usize>),
    Dash(Range<usize>),
    Integer(Range<usize>),
    IntegerDot(Range<usize>),
    Decimal(Range<usize>),
    Operator(Range<usize>),
    Illegal(Range<usize>),
}

impl Token {
    fn continued(&self, next_char: char) -> Option<Token> {
        let range = self.get().start..(self.get().end + next_char.len_utf8());
        match self {
            Token::Empty(_) => match next_char {
                ' ' | '\t' | '\n' | '\r' => Some(Token::Whitespace(range)),
                '-' => Some(Token::Dash(range)),
                '0'..='9' => Some(Token::Integer(range)),
                '.' => Some(Token::IntegerDot(range)),
                '(' => Some(Token::OpenParen(range)),
                ')' => Some(Token::CloseParen(range)),
                'd' | 'k' | 'K' | 'x' | 'X' | '^' | '*' | '/' | '+' | ',' => {
                    Some(Token::Operator(range))
                }
                _ => Some(Token::Illegal(range)),
            },
            Token::Whitespace(_) => match next_char {
                ' ' | '\t' | '\n' | '\r' => Some(Token::Whitespace(range)),
                _ => None,
            },
            Token::OpenParen(_) => None,
            Token::CloseParen(_) => None,
            Token::Dash(_) => match next_char {
                '0'..='9' => Some(Token::Integer(range)),
                _ => None,
            },
            Token::Integer(_) => match next_char {
                '0'..='9' => Some(Token::Integer(range)),
                '.' => Some(Token::IntegerDot(range)),
                _ => None,
            },
            Token::IntegerDot(_) => match next_char {
                '0'..='9' => Some(Token::Decimal(range)),
                _ => None,
            },
            Token::Decimal(_) => match next_char {
                '0'..='9' => Some(Token::Decimal(range)),
                _ => None,
            },
            Token::Operator(_) => None,
            Token::Illegal(_) => Some(Token::Illegal(range)),
        }
    }

    fn get(&self) -> Range<usize> {
        match self {
            Token::Empty(x) => *x..*x,
            Token::Whitespace(x)
            | Token::OpenParen(x)
            | Token::CloseParen(x)
            | Token::Dash(x)
            | Token::Integer(x)
            | Token::IntegerDot(x)
            | Token::Decimal(x)
            | Token::Operator(x)
            | Token::Illegal(x) => x.clone(),
        }
    }

    #[allow(unused)]
    fn get_str<'b>(&self, str: &'b str) -> &'b str {
        &str[self.get().clone()]
    }
}

impl<'a> Iterator for Lex<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if self.str.len() == self.start_offset {
            return None;
        }

        let mut token = Token::Empty(self.start_offset);
        for char in self.str[self.start_offset..].chars() {
            if let Some(next_token) = token.continued(char) {
                token = next_token;
                self.start_offset += char.len_utf8();
            } else {
                break;
            }
        }
        Some(token)
    }
}
fn lex(str: &str) -> impl Iterator<Item = Token> {
    Lex {
        str,
        start_offset: 0,
    }
}

#[derive(Debug, Clone)]
enum Value {
    Number(Number),
    #[allow(unused)]
    NumberList(Vec<Number>),
}
impl Value {
    fn as_number(&self) -> f64 {
        match self {
            Value::Number(x) => *x,
            Value::NumberList(items) => items.iter().sum(),
        }
    }
    fn as_num_list(&self) -> Vec<Number> {
        match self {
            Value::Number(x) => vec![*x],
            Value::NumberList(items) => items.clone(),
        }
    }
}

#[derive(Debug)]
enum ElementData {
    #[allow(unused)]
    Value(Value),
    UnaryOperator(char, Option<usize>),
    BinaryOperator(char, usize, Option<usize>),
    CloseParen,
}

#[derive(Debug)]
struct AstElement {
    position: usize,
    up: Option<usize>,
    data: ElementData,
}
impl AstElement {
    fn get_right(&mut self) -> Option<&mut Option<usize>> {
        match &mut self.data {
            ElementData::Value(_) | ElementData::CloseParen => None,
            ElementData::UnaryOperator(_, x) | ElementData::BinaryOperator(_, _, x) => Some(x),
        }
    }

    fn get_prerequisites(&self) -> [Option<usize>; 2] {
        match &self.data {
            ElementData::Value(_) => [None; 2],
            ElementData::UnaryOperator(_, prereq_a) => [*prereq_a, None],
            ElementData::BinaryOperator(_, prereq_a, prereq_b) => [Some(*prereq_a), *prereq_b],
            ElementData::CloseParen => [self.up, None],
        }
    }
}

#[derive(Debug)]
struct Ast {
    top_elem: Option<usize>,
    elements: Vec<AstElement>,
}
impl Ast {
    fn execute(&self) -> Result<Value, Error> {
        let Some(root) = self.top_elem else {
            return Ok(Value::Number(0.0));
        };
        let mut cache = vec![None; self.elements.len()];
        let mut stack = vec![root];
        loop {
            assert!(stack.len() <= cache.len());

            let Some(&top) = stack.last() else {
                #[cfg(test)]
                println!("{cache:?}");
                return cache.swap_remove(root).ok_or(Error {
                    error_type: ErrorType::Internal,
                    position: root,
                });
            };

            if cache[top].is_none() {
                let prereqs = self.elements[top].get_prerequisites();
                let mut are_all_prereqs_finished = true;
                for prereq in prereqs.into_iter().filter_map(|x| x) {
                    if cache[prereq].is_none() {
                        stack.push(prereq);
                        are_all_prereqs_finished = false;
                    }
                }
                if are_all_prereqs_finished {
                    #[cfg(test)]
                    println!("Calculating index {top}");

                    match self.calculate(&mut cache, top) {
                        Ok(x) => cache[top] = Some(x),
                        Err(Error {
                            error_type,
                            position: token_index,
                        }) => {
                            return Err(Error {
                                error_type,
                                position: self.elements[token_index].position,
                            });
                        }
                    }

                    stack.pop();
                } else {
                    #[cfg(test)]
                    println!("Deferring index {top}")
                }
            }
        }
    }

    fn calculate(&self, cache: &mut Vec<Option<Value>>, index: usize) -> Result<Value, Error> {
        let item = &self.elements[index].data;
        match item {
            ElementData::Value(value) => Ok(value.clone()),
            ElementData::UnaryOperator(op_char, param_idx) => {
                let Some(param_idx) = param_idx else {
                    return Err(Error {
                        error_type: ErrorType::Internal,
                        position: index,
                    });
                };
                let param_idx = *param_idx;

                let Some(param) = &cache[param_idx] else {
                    return Err(Error {
                        error_type: ErrorType::Internal,
                        position: index,
                    });
                };

                Ok(match op_char {
                    '-' => Value::Number(-param.as_number()),
                    '+' => Value::Number(param.as_number()),
                    'd' => {
                        let sides_f = param.as_number().floor();
                        if sides_f < 1.0 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: param_idx,
                            });
                        }
                        Value::NumberList(roll(1, sides_f as usize))
                    },
                    '(' => param.clone(),
                    _ => {
                        return Err(Error {
                            error_type: ErrorType::Internal,
                            position: index,
                        });
                    }
                })
            }
            ElementData::BinaryOperator(op_char, lparam_idx, rparam_idx) => {
                let Some(rparam_idx) = rparam_idx else {
                    return Err(Error {
                        error_type: ErrorType::Internal,
                        position: index,
                    });
                };

                let Some(lparam) = &cache[*lparam_idx] else {
                    return Err(Error {
                        error_type: ErrorType::Internal,
                        position: index,
                    });
                };
                let Some(rparam) = &cache[*rparam_idx] else {
                    return Err(Error {
                        error_type: ErrorType::Internal,
                        position: index,
                    });
                };

                Ok(match op_char {
                    'd' => {
                        let count_f = lparam.as_number().floor();
                        if count_f < 0.0 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: *lparam_idx,
                            });
                        }
                        let sides_f = rparam.as_number().floor();
                        if sides_f < 1.0 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: *rparam_idx,
                            });
                        }
                        Value::NumberList(roll(count_f as usize, sides_f as usize))
                    }
                    'k' => {
                        let mut l = lparam.as_num_list();
                        l.sort_unstable_by(|l, r| l.total_cmp(r));

                        let keep_amount_f = rparam.as_number().floor();
                        if keep_amount_f < 0.0 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: *rparam_idx,
                            });
                        }
                        l.truncate(keep_amount_f as usize);
                        Value::NumberList(l)
                    }
                    'K' => {
                        let mut l = lparam.as_num_list();
                        l.sort_unstable_by(|l, r| l.total_cmp(r).reverse());

                        let keep_amount_f = rparam.as_number().floor();
                        if keep_amount_f < 0.0 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: *rparam_idx,
                            });
                        }
                        l.truncate(keep_amount_f as usize);
                        Value::NumberList(l)
                    }
                    'x' => {
                        let mut l = lparam.as_num_list();
                        l.sort_unstable_by(|l, r| l.total_cmp(r).reverse());

                        let drop_amount_f = rparam.as_number().floor();
                        if drop_amount_f < 0.0 || drop_amount_f > l.len() as f64 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: *rparam_idx,
                            });
                        }
                        let keep_amount = l.len() - (drop_amount_f as usize);
                        l.truncate(keep_amount);
                        Value::NumberList(l)
                    }
                    'X' => {
                        let mut l = lparam.as_num_list();
                        l.sort_unstable_by(|l, r| l.total_cmp(r));

                        let drop_amount_f = rparam.as_number().floor();
                        if drop_amount_f < 0.0 || drop_amount_f > l.len() as f64 {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: *rparam_idx,
                            });
                        }
                        let keep_amount = l.len() - (drop_amount_f as usize);
                        l.truncate(keep_amount);
                        Value::NumberList(l)
                    }
                    '^' => {
                        let lhs = lparam.as_number();
                        let rhs = rparam.as_number();
                        let result = lhs.powf(rhs);
                        if !result.is_finite() {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: index,
                            });
                        }
                        Value::Number(result)
                    }
                    '*' => {
                        let lhs = lparam.as_number();
                        let rhs = rparam.as_number();
                        let result = lhs * rhs;
                        if !result.is_finite() {
                            return Err(Error {
                                error_type: ErrorType::BadSemantics,
                                position: index,
                            });
                        }
                        Value::Number(result)
                    }
                    '/' => {
                        let lhs = lparam.as_number();
                        let rhs = rparam.as_number();
                        let result = lhs / rhs;
                        if !result.is_finite() {
                            return Err(Error { error_type: ErrorType::BadSemantics, position: index })
                        }
                        Value::Number(result)
                    }
                    '+' => Value::Number(lparam.as_number() + rparam.as_number()),
                    '-' => Value::Number(lparam.as_number() - rparam.as_number()),
                    ',' => {
                        let l_len = match lparam {
                            Value::Number(_) => 1,
                            Value::NumberList(items) => items.len(),
                        };
                        let r_len = match rparam {
                            Value::Number(_) => 1,
                            Value::NumberList(items) => items.len(),
                        };
                        let mut result = Vec::with_capacity(l_len + r_len);
                        match lparam {
                            Value::Number(x) => result.push(*x),
                            Value::NumberList(items) => result.extend(items),
                        }
                        match rparam {
                            Value::Number(x) => result.push(*x),
                            Value::NumberList(items) => result.extend(items),
                        }
                        Value::NumberList(result)
                    },
                    _ => {
                        return Err(Error {
                            error_type: ErrorType::Internal,
                            position: index,
                        });
                    }
                })
            }
            ElementData::CloseParen => {
                let Some(up) = self.elements[index].up else {
                    return Err(Error {
                        error_type: ErrorType::Internal,
                        position: index,
                    });
                };
                self.calculate(cache, up)
            }
        }
    }
}

#[derive(Debug)]
struct AstBuilder {
    ast: Ast,
    open_paren_stack: Vec<usize>,
}
impl AstBuilder {
    fn add_token(&mut self, source_string: &str, token: Token) -> Result<(), Error> {
        let idx = self.ast.elements.len();
        match token {
            Token::Illegal(range) => {
                return Err(Error {
                    error_type: ErrorType::IllegalToken,
                    position: range.start,
                });
            }
            Token::Empty(_) => Ok(()),
            Token::Whitespace(_) => Ok(()),
            Token::OpenParen(range) => {
                self.open_paren_stack.push(idx);
                self.add_unary(range.start, '(')
            }
            Token::CloseParen(range) => self.add_close_paren(range.start),
            Token::Integer(range) | Token::IntegerDot(range) | Token::Decimal(range) => self
                .add_number_literal(
                    range.start,
                    source_string[range]
                        .parse::<f64>()
                        .expect("the token parsing should have forced this to be valid"),
                ),
            Token::Dash(range) | Token::Operator(range) => {
                let start = range.start;
                let op_string = &source_string[range];

                if let op @ ("+" | "-" | "d") = op_string {
                    // Consider the option of a unary operator
                    if self.can_accept_value_like() {
                        return self.add_unary(
                            start,
                            op.chars()
                                .next()
                                .expect("op is defined as a string with a length of at least 1"),
                        );
                    }
                }
                self.add_binary_operator(
                    start,
                    op_string
                        .chars()
                        .next()
                        .expect("A non-Token::Empty token must have at least one character"),
                )
            }
        }
    }

    fn add_unary(&mut self, err_index: usize, op_char: char) -> Result<(), Error> {
        self.add_value(err_index, ElementData::UnaryOperator(op_char, None))
    }

    fn add_number_literal(&mut self, err_index: usize, number: Number) -> Result<(), Error> {
        self.add_value(err_index, ElementData::Value(Value::Number(number)))
    }

    fn add_value(&mut self, err_index: usize, element_data: ElementData) -> Result<(), Error> {
        assert!(matches!(
            element_data,
            ElementData::Value(_) | ElementData::UnaryOperator(_, _)
        ));

        let add_pos = self.ast.elements.len();
        if self.can_accept_value_like() {
            let l = self.ast.elements.len();
            let (up_backref, up) = if let Some(last_op) = self.ast.elements.last_mut() {
                (
                    last_op
                        .get_right()
                        .expect("can_accept_value_like implies get_right is Some"),
                    Some(l - 1),
                )
            } else {
                (&mut self.ast.top_elem, None)
            };
            *up_backref = Some(add_pos);

            self.ast.elements.push(AstElement {
                position: err_index,
                up,
                data: element_data,
            });
            Ok(())
        } else {
            Err(Error {
                error_type: ErrorType::ExpectedNonValue,
                position: err_index,
            })
        }
    }

    fn add_close_paren(&mut self, err_index: usize) -> Result<(), Error> {
        if self.can_accept_value_like() {
            return Err(Error {
                error_type: ErrorType::ExpectedValueGotParen,
                position: err_index,
            });
        }

        let Some(matching_paren) = self.open_paren_stack.pop() else {
            return Err(Error {
                error_type: ErrorType::UnmatchedParen,
                position: err_index,
            });
        };

        self.ast.elements.push(AstElement {
            position: err_index,
            up: Some(matching_paren),
            data: ElementData::CloseParen,
        });
        Ok(())
    }

    fn add_binary_operator(&mut self, err_index: usize, op_char: char) -> Result<(), Error> {
        if self.can_accept_value_like() {
            return Err(Error {
                error_type: ErrorType::ExpectedValueGotBinaryOperator,
                position: err_index,
            });
        }

        let add_idx = self.ast.elements.len();

        let left_idx = self.ast.elements.len() - 1;

        // Currently, it's invalid to add this operator because the left_idx doesn't point to a thing which `up`` points to this

        let mut operator = ElementData::BinaryOperator(op_char, left_idx, None);

        let mut steal_from_idx = self.ast.elements[left_idx].up;
        while let Some(steal_idx) = steal_from_idx {
            if self.can_steal_from(&self.ast.elements[steal_idx].data, &operator) {
                break;
            }

            steal_from_idx = self.ast.elements[steal_idx].up;
        }
        // Either steal_from_idx is None (and we need to steal from the AST's top_elem)
        // or steal_from_idx is Some(idx) and we steal from ast.elem[idx]

        let down_point_ref = steal_from_idx.map_or(&mut self.ast.top_elem, |x| {
            self.ast.elements[x].get_right().expect(
                "if going up leads to some element, that element should always have a right",
            )
        });

        let lower_idx = down_point_ref.unwrap();
        let ElementData::BinaryOperator(_, ref mut operator_left, _) = operator else {
            panic!("unreachable")
        };
        *operator_left = lower_idx;
        *down_point_ref = Some(add_idx);

        let upper_idx = self.ast.elements[lower_idx].up;

        self.ast.elements.push(AstElement {
            position: err_index,
            up: upper_idx,
            data: operator,
        });
        self.ast.elements[lower_idx].up = Some(add_idx);

        Ok(())
    }

    fn can_steal_from(&self, left: &ElementData, right: &ElementData) -> bool {
        let precedence = |x: &ElementData| match x {
            ElementData::BinaryOperator(op_char, _, _) | ElementData::UnaryOperator(op_char, _) => {
                match op_char {
                    'd' => 0,
                    'k' | 'K' | 'x' | 'X' => 1,
                    '^' => 2,
                    '*' | '/' => 3,
                    '+' | '-' => 4,
                    ',' => 5,
                    _ => u8::MAX,
                }
            }
            _ => u8::MAX,
        };
        let lpreceedence = precedence(left);
        let rpreceedence = precedence(right);

        lpreceedence > rpreceedence || (lpreceedence == 2 && rpreceedence == 2)
    }

    fn can_accept_value_like(&self) -> bool {
        matches!(
            self.ast.elements.last(),
            None | Some(AstElement {
                position: _,
                up: _,
                data: ElementData::BinaryOperator(_, _, _) | ElementData::UnaryOperator(_, _)
            })
        )
    }

    fn finalize(self) -> Result<Ast, Error> {
        if let Some(&unmatched_idx) = self.open_paren_stack.last() {
            return Err(Error {
                error_type: ErrorType::UnmatchedParen,
                position: unmatched_idx,
            });
        }
        Ok(self.ast)
    }
}

fn parse<'a>(source_string: &str, tokens: impl IntoIterator<Item = Token>) -> Result<Ast, Error> {
    let mut ast_builder = AstBuilder {
        ast: Ast {
            elements: Vec::new(),
            top_elem: None,
        },
        open_paren_stack: Vec::new(),
    };
    for token in tokens {
        ast_builder.add_token(source_string, token)?;

        #[cfg(test)]
        println!("{ast_builder:#?}");
    }

    ast_builder.finalize()
}

#[cfg(test)]
mod test;
