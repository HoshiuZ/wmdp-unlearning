"""Exclusive offline system outcome; labels and experiment-defined policies remain separate."""
def outcome(input_audit,generation,output_audit,policy):
    decision=input_audit['policies'][policy]
    if decision['error_unknown']:return 'exception_input'
    if decision['blocked']:return 'input_block'
    if generation is None or generation['status']!='ok':return 'exception_generation'
    if output_audit is None:return 'exception_output'
    decision=output_audit['policies'][policy]
    if decision['error_unknown']:return 'exception_output'
    if decision['blocked']:return 'output_block'
    return 'allow'
