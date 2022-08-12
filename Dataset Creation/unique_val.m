function new_array = unique_val(array, step) 
    new_array=array;
    while(length(new_array) ~= length(unique(new_array)))
        for i=1:length(new_array)
            k = find(new_array == new_array(i));
            if length(k)>1
                for j=2:length(k)
                    new_array(k(j)) = new_array(k(j)) + (j-1)*step;
                end
            end
        end
    end
end