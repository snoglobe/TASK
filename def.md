# the TASK format
a message format for LLMs that is designed for performing tasks broken into steps and promotes long-horizon thinking
# literals:
## numbers
just normal numbers. all formats supported by JS are allowed (hex, octal, binary, whatever)
## strings
when a string is expected, if the string does not contain spaces then it is written without quotes. otherwise, for example a tool call with a string input, we use regular quotes ("). special quotes (「」) are also allowed for multi-line, long strings that might contain regular quotes (e.g. code).
## arrays
```js
[ a • b • c ]
```
## object syntax
brace-delimited, json like
### example object: 
```js
{ a ↦ b • c ↦ d } - {"a":"b", "c":"d"}
```
## id syntax
everything (all expressions and top-level structures) can be "tagged" with a semantic tag for later reference. the post-fix operator 🏷 does this.
### references
the post-fix operator ※ is used to denote comma-seperated references to tags made with the 🏷 post-fix operator. ※ can be used on everything as well. 
## structures
structures are the top-level pieces of the turn-based message system
### structure types
#### tool : a tool definition
```js
tool {
    name ↦ get_weather •
    params ↦ {
        zip_code ↦ {
            type ↦ string 
        } •
        unit ↦ {
            enum ↦ [
                metric • 
                imperial
            ] •
            required ↦ false
        }
    }
}
```
##### notes:
- required is implicitly true
- type is not required for enum values
- supports everything the OpenAI tool call specification does

#### system : a system message
```js
system「You are a helpful assistant.」🏷 sys1
```
##### notes:
- we use special quotes instead of regular quotes here because the message is more likely to contain "" than 「」.
#### user : a user message
```js
user「What is the weather in 94103 today?」🏷 usr1
```
#### todo : the internal todo list
the todo list is an internal structure that the model maintains to assist in task completion. 
the structure includes it as an approach to increasing task follow-through in models by teaching a step-oriented approach to problem solving, "baked in" to the model. 
```js
todo {
    1 ↦ "Understand the codebase." •
    2 ↦ "Identify where the problem is." •
    3 ↦ "Identify a possible solution." •
    4 ↦ "Implement the solution."
}
```
##### notes:
- the model maintains this list itself - it is not mutated or injected by the inference scaffolding. it is a fully internal mechanism designed to promote long-horizon task completion. 
##### the satisfies post-fix operator:
- the ⊨ post-fix operator is used on any expression that satisfies a to-do list item. 
```js
...

    todo {
        1 ↦ "Fetch the weather information." •
        2 ↦ "Determine the best clothing for the weather." ※ usr1 •
        3 ↦ "Present to the user."
    }

...

    call {
        tool ↦ get_weather •
        zip_code ↦ "94103" •
        id ↦ "weather-result"
    } ⊨ 1
...
```
#### plan : the model's planning phase
a turn by the model meant for planning. this is where it creates the todo list
```js
plan {
    todo ↦ {
        1 ↦ "Fetch the weather information." •
        2 ↦ "Determine the best clothing for the weather." ※ usr1 •
        3 ↦ "Present to the user."
    } •
    rationale ↦ "The user wants to know the weather, and the get_weather tool will give us live weather data. Then, they want to know the best clothing to wear. I will consider what the best clothing for the weather is given the output of the get_weather tool, and finally present it to the user."
}
```
#### act : the model's action phase
the model can either call tools (can call multiple using an array)
```js
act {
    call ↦ {
        tool ↦ get_weather •
        zip_code ↦ "94103" •
        id ↦ "weather-result"
    } ⊨ 1
}
```
gets the result (inserted by the inference system after the tool result is given from the client):
```js
result {
    data ↦ "The weather at 94103 is 68 degrees Fahrenheit, and overcast."
} 🏷 "weather-result"
```
ponders the clothing:
```js
act {
    think ↦ "Given that the weather is 68 degrees Fahrenheit and overcast, it seems that a light sweater, a t-shirt, and sweatpants would make good clothes for today." 𝑝 0.9 ※ "weather-result" 🏷 rationale ⊨ 2
}
```
##### the confidence post-fix operator
the 𝑝 operator goes on the end of any statements that the model makes that it is fitting to assign a confidence value to. it is between 0.0 and 1.0. the model may be 1.0 confident about things that are pure signal; i.e. tool results (assumed correct), or things that are blatantly true.
##### notes:
- the model can think and call at the same time, and call multiple tools.
```js
act {
    think ↦ "Pondering tool calls and args..." 🏷 thinky •
    call ↦ {
        tool ↦ foo •
        bar ↦ baz ※ thinky • 
        id ↦ bong
    } •
    call ↦ {
        tool ↦ bar ※ thonky • 
        baz ↦ bong •
        id ↦ bing
    } 
}

[inserted by provider]:
result {
    data ↦ "Response 1"
} 🏷 bong

result {
    data ↦ "Response 2"
} 🏷 bing
```
#### response : the model's response
```js
response「The weather in 94103 today is 68 degrees Fahrenheit and overcast. Today would be a good day for a light sweater, a t-shirt, and sweatpants.」※ ["weather-result" • rationale] ⊨ 3
```
## a whole trace
```js
system「You are a helpful assistant.」🏷 sys1

tool {
    name ↦ get_weather •
    params ↦ {
        zip_code ↦ {
            type ↦ string 
        } •
        unit ↦ {
            enum ↦ [
                metric • 
                imperial
            ] •
            required ↦ false
        }
    }
}

user「What is the weather in 94103 today?」🏷 usr1

plan {
    todo ↦ {
        1 ↦ "Fetch the weather information." •
        2 ↦ "Determine the best clothing for the weather." ※ usr1 •
        3 ↦ "Present to the user."
    } •
    rationale ↦ "The user wants to know the weather, and the get_weather tool will give us live weather data. Then, they want to know the best clothing to wear. I will consider what the best clothing for the weather is given the output of the get_weather tool, and finally present it to the user."
}

act {
    call ↦ {
        tool ↦ get_weather •
        zip_code ↦ "94103" •
        id ↦ "weather-result"
    } ⊨ 1
}

result {
    data ↦ "The weather at 94103 is 68 degrees Fahrenheit, and overcast."
} 🏷 "weather-result"

act {
    think ↦ "Given that the weather is 68 degrees Fahrenheit and overcast, it seems that a light sweater, a t-shirt, and sweatpants would make good clothes for today." 𝑝 0.9 ※ "weather-result" 🏷 rationale ⊨ 2
}

response「The weather in 94103 today is 68 degrees Fahrenheit and overcast. Today would be a good day for a light sweater, a t-shirt, and sweatpants.」※ ["weather-result" • rationale] ⊨ 3
```
### notes:
- this can keep going beyond this example trace
- the sequence must always go like this:

sys_msg

tools

usr_msg

plan 

(any combo of act/plan, preferably without more than one plan in a row)

response (user can send follow-ups and continue the cycle from usr_msg down)

- user message IDs should always be in the format `usrN` where `N` is the nth user message
