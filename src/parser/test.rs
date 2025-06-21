use super::{Token, lex, parse};

#[test]
fn test_lex() {
    let lexstr = "1d8 + -4";
    let result: Vec<_> = lex(lexstr).collect();
    assert_eq!(
        result,
        vec![
            Token::Integer(0..1),
            Token::Operator(1..2),
            Token::Integer(2..3),
            Token::Whitespace(3..4),
            Token::Operator(4..5),
            Token::Whitespace(5..6),
            Token::Integer(6..8),
        ]
    );
}

#[test]
fn test_parse() {
    let parsestr = "1d8 + -4";
    let tokens = lex(parsestr);
    let ast = parse(parsestr, tokens).unwrap();
    println!("{ast:#?}");
}

#[test]
fn test_exec() {
    let parsestr = "1d8 + -4";
    let tokens = lex(parsestr);
    let ast = parse(parsestr, tokens).unwrap();
    println!("{ast:#?}");
    for _ in 0..10 {
        let result = ast.execute();
        println!("Rolled {result:?}");
    }
}

#[test]
fn test_multiple_add() {
    let parsestr = "1+2+3";
    let ast = parse(parsestr, lex(parsestr)).unwrap();
    println!("{ast:#?}");
    assert_eq!(ast.execute().unwrap().as_number(), 6.);
}

#[test]
fn test_right_associativity() {
    let parsestr = "2^2^2";
    let ast = parse(parsestr, lex(parsestr)).unwrap();
    println!("{ast:#?}");
    assert_eq!(ast.execute().unwrap().as_number(), 16.);
}

#[test]
fn test_paren() {
    let parsestr = "(1+2)";
    let ast = parse(parsestr, lex(parsestr)).unwrap();
    println!("{ast:#?}");
    assert_eq!(ast.execute().unwrap().as_number(), 3.);
}
