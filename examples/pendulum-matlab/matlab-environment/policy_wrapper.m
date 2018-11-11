function u = policy_wrapper(mu,s)
    global policy
    u = policy.fcn(policy,mu,s);
end

