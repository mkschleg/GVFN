

module Experiment

using Reproduce

abstract type AbstractExperiment end
abstract type AbstractRLExperiment <: AbstractExperiment end



function arg_settings(as::ArgParseSettings =
                      ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler),
                      exp::AbstractRLExperiment)
end

function save_directory(exp::AbstractRLExperiment)
end





end


