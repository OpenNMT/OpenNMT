-- adapted from https://github.com/bgshih/crnn/blob/4179a87c54cfe4f57b49a34a9871a4a5750bbf34/src/utilities.lua
require 'class'
require 'paths'

logging = torch.class('logger')

function logging:__init(log_path)
    local open_mode = 'w'
    if paths.filep(log_path) then
        local input = nil
        while not input do
            print('Logging file exits. Overwrite(o)? Append(a)? Abort(q)?')
            input = io.read()
            if input == 'o' or input == 'O' then
                open_mode = 'w'
            elseif input == 'a' or input == 'A' then
                open_mode = 'a'
            elseif input == 'q' or input == 'Q' then
                os.exit()
            else
                open_mode = 'a'
            end
        end
    end
    self.logger_file = io.open(log_path, open_mode)
end

function logging:info(message, mute)
    mute = mute or false
    local time_stamp = os.date('%x %X')
    local msg_formatted = string.format('[%s]  %s', time_stamp, message)
    if not mute then
        print (msg_formatted)
    end
    if self.logger_file then
        self.logger_file:write(msg_formatted .. '\n')
        self.logger_file:flush()
    end
end

function logging:shutdown()
    if self.logger_file then
        self.logger_file:close()
    end
end
