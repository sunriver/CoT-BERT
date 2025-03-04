
# from syntactic_complexity import analyze_sentence
def compute_complexity(sentence):
    # sentence = "The quick brown fox jumps over the lazy dog."
    # complexity = analyze_sentence(sentence)
    # print(f"Syntactic Complexity Score: {complexity}")
    return 1.0  # 仅保留句法复杂度惩罚项

MAX_LENGTH = 50
def _compute_weights(length, syntax_complexity):
    alpha = 1.0 - (length / MAX_LENGTH) * 0.5
    beta = 1.0 - syntax_complexity * 0.2  # 仅保留句法复杂度惩罚项
    return alpha, beta

def compute_weights(sentence):
    sent_len = len(sentence)
    sent_complexity =  syntactic_complexity(sentence)
    alpha, beta = _compute_weights(sent_len, sent_complexity)
    return alpha, beta


def main():
    # 示例
    # result = analyze_dependencies("Apple decided to buy a startup in London.")
    # print(f"句子: {result['sentence']}")
    # print(f"唯一依存关系类型数量: {result['unique_dependencies']}")  # 输出: 5
    # print(f"平均复杂度（子节点数）: {result['average_complexity']:.2f}")  # 输出: 1.67

    alpha, beta = compute_weights("Apple decided to buy a startup in London.")
    print(f"alpha: {alpha}, beta: {beta}")  # 输出: alpha: 0.5, beta: 0.8

if __name__ == "__main__":
    main()
