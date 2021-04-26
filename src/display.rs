
///
/// Creates a string representation of a confusion matrix with the provided counts and class names
///
/// # Arguments
///
/// * `counts` the confusion matrix counts
/// * `classes` the class outcome names
///
pub fn stringify(counts: &Vec<Vec<usize>>, classes: &Vec<String>) -> String {
    // Compute max length of outcome names (in chars)
    let max_class_length = classes.iter().fold(0, |max, outcome| {
        match outcome.chars().count() {
            length if length > max => length,
            _ => max
        }
    });
    // Two spaces on either side, plus a leading pipe character
    let padded_class_length = max_class_length + 5;
    // Two spaces on either side of "Prediction", plus a leading pipe character
    let prediction_wing_length = padded_class_length + 15;
    // Build the output string
    let mut output = String::new();
    write_cm_top_rows(classes, prediction_wing_length, padded_class_length, &mut output);
    write_cm_data_rows(counts, classes, prediction_wing_length, padded_class_length, &mut output);
    output
}

fn write_cm_top_rows(outcomes: &Vec<String>,
                     prediction_wing_length: usize,
                     padded_outcome_length: usize,
                     buffer: &mut String) {
    // 1st row
    fill_char(' ', prediction_wing_length, buffer);
    buffer.push('o');
    fill_char('=', outcomes.len() * padded_outcome_length - 1, buffer);
    buffer.push_str("o\n");

    // 2nd row
    fill_char(' ', prediction_wing_length, buffer);
    buffer.push('|');
    buffer.push_str(center("Label", (outcomes.len() * padded_outcome_length - 1)).as_str());
    buffer.push_str("|\n");

    // 3rd row
    fill_char(' ', prediction_wing_length, buffer);
    buffer.push('|');
    fill_char('=', outcomes.len() * padded_outcome_length - 1, buffer);
    buffer.push_str("|\n");

    // 4th row
    fill_char(' ', prediction_wing_length, buffer);
    buffer.push('|');
    for i in 0..outcomes.len() {
        let content = center(outcomes[i].as_str(), padded_outcome_length - 1);
        buffer.push_str(format!("{}|", content).as_str());
    }
    buffer.push('\n');

    // 5th row
    buffer.push('o');
    fill_char('=', prediction_wing_length - 1, buffer);
    buffer.push('o');
    for _ in 1..outcomes.len() {
        fill_char('=', padded_outcome_length - 1, buffer);
        buffer.push('|');
    }
    fill_char('=', padded_outcome_length - 1, buffer);
    buffer.push_str("o\n");
}

fn write_cm_data_rows(counts: &Vec<Vec<usize>>,
                      outcomes: &Vec<String>,
                      prediction_wing_length: usize,
                      padded_outcome_length: usize,
                      buffer: &mut String) {

    for (i, outcome) in outcomes.iter().enumerate() {
        buffer.push('|');
        if i == outcomes.len() / 2 && outcomes.len() % 2 != 0 {
            buffer.push_str("  Prediction  |");
        } else {
            buffer.push_str("              |");
        }
        buffer.push_str(center(outcome.as_str(), padded_outcome_length - 1).as_str());
        buffer.push('|');
        for j in 0..outcomes.len() {
            let fmt_count = center(counts[i][j].to_string().as_str(), padded_outcome_length - 1);
            buffer.push_str(format!("{}|", fmt_count).as_str());
        }
        buffer.push('\n');
        if i < outcomes.len() - 1 {
            if i == outcomes.len() / 2 - 1 && outcomes.len() % 2 == 0 {
                buffer.push_str("|  Prediction  |");
            } else {
                buffer.push_str("|              |");
            }
            fill_char('=', padded_outcome_length - 1, buffer);
            buffer.push('|');
            for _ in 0..outcomes.len() {
                fill_char('-', padded_outcome_length - 1, buffer);
                buffer.push('|');
            }
            buffer.push('\n');
        } else {
            buffer.push('o');
            fill_char('=', prediction_wing_length - 1, buffer);
            buffer.push('o');
            fill_char('=', padded_outcome_length * outcomes.len() - 1, buffer);
            buffer.push('o');
        }
    }
}

fn fill_char(c: char, length: usize, buffer: &mut String) {
    for _ in 0..length {
        buffer.push(c);
    }
}

fn center(s: &str, length: usize) -> String {
    let len: usize = s.chars().count();
    let diff = length - len;
    let left_pad_length = diff / 2;
    let right_pad_length = diff - left_pad_length;
    let left_pad: String = vec![' '; left_pad_length].iter().collect();
    let right_pad: String = vec![' '; right_pad_length].iter().collect();
    format!("{}{}{}", left_pad, s, right_pad)
}
