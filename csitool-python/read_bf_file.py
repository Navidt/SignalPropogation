def read_bf_file(filename: str):
  # Open file
  f = open(filename, 'rb')
  if f.fileno < 0:
    raise Exception(f"Couldn''t open file {filename}")

  # go to the end of file
  status = f.seek(0, 2)
  if status != 0:
      f.close()
      raise Exception("Error seeking to eof")
  
  len = f.tell()

  # go to the beginning of the file
  status = f.seek(0, 0)
  if status != 0:
      f.close()
      raise Exception("Error seeking to bof")

  # Initialize variables
  ret = cell(ceil(len/95),1);     % Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
  cur = 0;                        % Current offset into file
  count = 0;                      % Number of records output
  broken_perm = 0;                % Flag marking whether we've encountered a broken CSI yet
  triangle = [1 3 6];             % What perm should sum to for 1,2,3 antennas

  %% Process all entries in file
  % Need 3 bytes -- 2 byte size field and 1 byte code
  while cur < (len - 3)
      % Read size and code
      field_len = fread(f, 1, 'uint16', 0, 'ieee-be');
      code = fread(f,1);
      cur = cur+3;
      
      % If unhandled code, skip (seek over) the record and continue
      if (code == 187) % get beamforming or phy data
          bytes = fread(f, field_len-1, 'uint8=>uint8');
          cur = cur + field_len - 1;
          if (length(bytes) ~= field_len-1)
              fclose(f);
              return;
          end
      else % skip all other info
          fseek(f, field_len - 1, 'cof');
          cur = cur + field_len - 1;
          continue;
      end
      
      if (code == 187) %hex2dec('bb')) Beamforming matrix -- output a record
          count = count + 1;
          ret{count} = read_bfee(bytes);
          
          perm = ret{count}.perm;
          Nrx = ret{count}.Nrx;
          if Nrx == 1 % No permuting needed for only 1 antenna
              continue;
          end
          if sum(perm) ~= triangle(Nrx) % matrix does not contain default values
              if broken_perm == 0
                  broken_perm = 1;
                  fprintf('WARN ONCE: Found CSI (%s) with Nrx=%d and invalid perm=[%s]\n', filename, Nrx, int2str(perm));
              end
          else
              ret{count}.csi(:,perm(1:Nrx),:) = ret{count}.csi(:,1:Nrx,:);
          end
      end
  end
  ret = ret(1:count);

  %% Close file
  fclose(f);
  end