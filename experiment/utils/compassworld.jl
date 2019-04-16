module CycleWorldUtils

function env_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--size"
        help="The size of the compass world"
        arg_type=Int64
        default=8
    end
end






end
