class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res = 0
        left = 0
        right = len(height)-1
        while left < right:
            l = height[left]
            r = height[right]
            temp = min(l, r) * (right-left)
            if temp > res:
                res = temp
            if l < r:
                left += 1
            else:
                right -= 1
        return res
    
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height)-1
        left_max = 0
        right_max = 0
        res = 0
        while left < right:
            if height[left] <= height[right]:
                #go left
                if height[left] > left_max:
                    left_max = height[left]
                else:
                    res += (left_max-height[left])
                left += 1
            else:
                #go right
                if height[right] > right_max:
                    right_max = height[right]
                else:
                    res += (right_max-height[right])
                right -= 1


        return res
    
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # space
        # time
        if len(nums) < 3:
            return []
        res = set()
        nums.sort()
        for i in range(0, len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                if nums[i]+nums[j]+nums[k]==0:
                    res.add((nums[i], nums[j], nums[k]))
                    j += 1
                    k -= 1
                    
                elif nums[i]+nums[j]+nums[k] < 0:
                    j+=1
                else:
                    k-=1

        return res
    
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # space
        # time
        res = float("inf")
        nums.sort()
        if len(nums) < 3:
            return 0
        for i in range(0, len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                temp = nums[i]+nums[j]+nums[k]
                if temp == target:
                    return target
                elif temp < target:
                    j+=1      
                else:
                    k-=1
                curr_diff = abs(temp-target)
                prev_diff = abs(res-target)

                if curr_diff < prev_diff:
                    res = temp
        return res
    
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = set()
        if len(nums) < 4:
            return []
        nums.sort()

        for a in range(0, len(nums)-3):
            for b in range(a+1, len(nums)-2):
                c = b+1
                d = len(nums)-1     
                while c < d:
                    temp = nums[a]+nums[b]+nums[c]+nums[d]
                    if temp == target:
                        res.add((nums[a], nums[b], nums[c], nums[d]))
                        c += 1
                        d -= 1
                    elif temp < target:
                        c += 1
                    else:
                        d -= 1
        return res
    
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero = 0
        one = 0
        two = len(nums)-1

        while one <= two:
            if nums[one] == 0:
                nums[one], nums[zero] = nums[zero], nums[one]
                zero += 1
                one += 1
            elif nums[one] == 1:
                one += 1
            else:
                nums[one], nums[two] = nums[two], nums[one]
                two -= 1

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left = 0
        right = len(s)-1
        
        while left <= right:
            l = ""
            r = ""
            while left <= len(s)-1 and not s[left].isalnum():
                left += 1
            if left <= len(s)-1:
                l = s[left].lower()
            while right >= 0 and not s[right].isalnum():
                right -= 1
            if right >= 0:
                r = s[right].lower()
            if l != r:
                return False
            left += 1
            right -=1
            

        return True
    
    def validPalindrome(self, s):
        left = 0
        right = len(s)-1

        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.check(s, left+1, right) or self.check(s, left, right-1)
        return True
    
    def check(self, s, left, right):
        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False
        return True
    
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        words = s.split()
        res = ""
        p = len(words)
        while p:
            if p == len(words):
                res += words[p-1]
            else:
                res += " " + words[p-1]
            p -= 1

        return res

    def reverseWords(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        print(s)
        
        start = 0
        end = 0
        while end <= len(s)-1:
            while end <= len(s)-1 and s[end] != " ":
                end += 1
            e = end -1
            while start <= e:
                s[start], s[e] = s[e], s[start]
                start += 1
                e -= 1
            start = end+1
            end += 1
        
        ee = len(s)-1
        while start <= ee:
            s[start], s[ee] = s[ee], s[start]
            start += 1
            ee -= 1

    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
        start = 0
        end = 0
        while end <= len(s)-1:
            while end <= len(s)-1 and s[end] != " ":
                end += 1
            e = end-1
            while start <= e:
                s[start], s[e] = s[e], s[start]
                start += 1
                e -= 1
            start = end+1
            end += 1
        
        e = len(s)-1
        while start <= e:
            s[start], s[e] = s[e], s[start]
            start += 1
            e -= 1
        
        return "".join(s)
    
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        seen = set()
        while head:
            if head in seen:
                return head
            seen.add(head)
            head = head.next
        return None
    
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False
    
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # left = 0
        # right = len(nums)-1

        # if nums[left] == nums[right]:
        #     return nums[left]
        # while left < right:
        #     middle = left+1
        #     while middle < right:
        #         if nums[middle] == nums[right] or nums[middle] == nums[left]:
        #             return nums[middle]
        #         middle += 1
        #     left += 1
        #     right -= 1

        s = nums[0]
        f = nums[0]

        while True:
            s = nums[s]
            f = nums[nums[f]]
            if s == f:
                break
        
        s = nums[0]

        while s != f:
            s = nums[s]
            f = nums[f]

        return s

    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        intervals.sort(key=lambda x:x[0])
        print(intervals)
        need = Queue.PriorityQueue()
        need.put(intervals[0][1])
        for i in range(1, len(intervals)):
            start = intervals[i][0]
            end = intervals[i][1]
            curr_end = need.get()
            if start < curr_end:
                need.put(curr_end)
            need.put(end)
        return need.qsize()

    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = head
        fast = head
        prev = None

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = slow
        slow = slow.next
        prev.next = None

        while slow:
            n = slow.next
            slow.next = prev
            prev = slow
            slow = n
        
        slow = prev
        fast = head

        while slow:
            if fast.val != slow.val :
                return False
            fast = fast.next
            slow = slow.next
        return True
    
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        last = m+n-1
        a = m-1
        b = n-1

        while b >= 0:
            if a >= 0 and nums1[a] > nums2[b]:
                nums1[last] = nums1[a]
                a -= 1
                
            else:
                nums1[last] = nums2[b]
                b -= 1
                
            last -= 1

    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        before_head = ListNode(0)
        before = before_head
        after_head = ListNode(0)
        after = after_head

        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        after.next = None
        before.next = after_head.next
        return before_head.next
    
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr = nums[0]
        idx = 1
        for i in range(1, len(nums)):
            if nums[i] == curr:
                continue
            nums[idx] = nums[i]
            curr = nums[i]
            idx +=1
        return idx
    
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen = set()
        while True:
            if n == 1:
                return True
            elif n in seen:
                return False
            else:
                seen.add(n)
                temp = 0
                for i in str(n):
                    temp += int(math.pow(int(i), 2))
                n = temp
    
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        #find peak
        for i in range(len(nums)-1, 0, -1):
            if nums[i - 1] < nums[i]:
                nums[i:] = sorted(nums[i:])

                pre_idx = i - 1
            
                for i in range(i, len(nums)):
                    if nums[pre_idx] < nums[i]:
                        nums[pre_idx], nums[i] = nums[i], nums[pre_idx]
                        return nums
        
        return nums.reverse()
    
    def circularArrayLoop(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        visit = set()
        for i in range(len(nums)):
            cycle = set()
            if i not in visit:
                while True:
                    if i in cycle:
                        return True
                    if i in visit:
                        break
                    visit.add(i)
                    cycle.add(i)
                    prev = i
                    i = (i+nums[i]) % len(nums)
                    if prev == i or ((nums[prev]>0) != (nums[i]>0)):
                        break
        return False
    
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) > len(s2):
            return False
        
        cnt1 = [0]*26
        cnt2 = [0]*26

        for i in range(len(s1)):
            cnt1[ord(s1[i])-ord('a')] += 1

        for i in range(len(s2)):
            cnt2[ord(s2[i])-ord('a')] += 1
            if i >= len(s1):
                cnt2[ord(s2[i-len(s1)])-ord('a')] -= 1

            if cnt1 == cnt2:
                return True

        return False

    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        res = []
        d = dict()
        for idx, char in enumerate(s):
            d[char] = idx
        
        start = 0
        end = 0
        for idx, char in enumerate(s):
            end = max(end, d[char])
            if end == idx:
                res.append(end-start+1)
                start = end+1
        
        return res
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0 
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res = max(res, prices[i]-buy)
        return res
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res += (prices[i] - buy)
                buy = prices[i]

        return res
    
    def totalFruit(self, fruits):
        """
        :type fruits: List[int]
        :rtype: int
        """
        res = 0
        d = collections.defaultdict(int)

        l = 0
        for r, fruit in enumerate(fruits):
            d[fruit] += 1
            while len(d) > 2:
                d[fruits[l]] -= 1
                if d[fruits[l]] == 0:
                    del d[fruits[l]]
                l += 1
            res = max(res, r-l+1)

        return res
    
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left = 0
        res = 0
        right = 0
        seen = dict()
        while right < len(s):
            if s[right] in seen:
                left = max(left, seen[s[right]]+1)
            res = max(res, right-left+1)
            seen[s[right]] = right
            right += 1
        return res
    
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res, d = [], {}
        for i in range(len(s)):
            if s[i:i+10] not in d: 
                d[s[i :i+10]] = 0
            else:
                if s[i:i+10] not in res: 
                    res.append(s[i:i+10])
        
        return res
    
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        left = 0
        right = len(arr) - 1

        while right-left >= k:
            if abs(x-arr[left]) <= abs(x-arr[right]):  
                right -= 1
            else:
                left += 1
        return arr[left:right+1]
    
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        if sum(nums) < target:
            return 0
        elif sum(nums) == target:
            return len(nums)

        res = float("inf")
        left = 0
        temp = nums[left]
        if temp >= target:
            res = 1
        for right in range(1, len(nums)):   
            temp += nums[right]
            if temp >= target:
                res = min(res, right-left+1)
                while temp >= target:
                    temp -= nums[left]
                    left += 1
                    if temp >= target:
                        res = min(res, right-left+1)
        return res

    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        q = collections.deque()
        res = []
        for i in range(len(nums)):
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if q[0] == i-k:
                q.popleft()
            if i >= k-1:
                res.append(nums[q[0]])
        return res
    
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if len(t) > len(s):
            return ""
        
        left = 0
        start = 0
        end = 0
        missing = len(t)
        need = collections.Counter(t)
        for right, char in enumerate(s, 1):
            if need[char] > 0:
                missing -= 1
            need[char] -= 1
            if missing == 0:
                while left < right and need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                need[s[left]] += 1
                missing += 1
                if end == 0 or right -left < end-start:
                    start = left
                    end = right
                left+=1
        return s[start:end]
    
    def findLength(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        n2 = "".join([chr(x) for x in nums2])
        curr = ""
        res = 0
        for num in nums1:
            curr += chr(num)
            if curr in n2:
                res = max(res, len(curr))
            else:
                curr = curr[1:]
        return res

    def countSubarrays(self, nums, minK, maxK):
        """
        :type nums: List[int]
        :type minK: int
        :type maxK: int
        :rtype: int
        """
        currMin = -1
        currMax = -1
        currOver = -1
        res = 0
        for i in range(len(nums)):
            if nums[i] == minK:
                currMin = i
            if nums[i] == maxK:
                currMax = i
            if nums[i] > maxK or nums[i] < minK:
                currOver = i
            if currMin != -1 and currMax != -1:
                start = min(currMin, currMax)
                if currOver < start:
                    res += start - currOver
        return res
    
    def maxTurbulenceSize(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        res = 0
        temp = 0
        for i in range(len(arr)):
            if i >= 2 and (arr[i-2] > arr[i-1] < arr[i] or arr[i-2] < arr[i-1] > arr[i]):
                    temp += 1
            elif i >= 1:
                if arr[i] != arr[i-1]:
                    temp = 2
            else:
                temp = 1
            res = max(res, temp)
        return res

    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        if not intervals:
            return True
        intervals.sort(key=lambda x:x[0])
        end = intervals[0][1]
        q = Queue.PriorityQueue()
        q.put(end)
        for i in range(1, len(intervals)):
            prev_end = q.get()
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if curr_start < prev_end:
                q.put(prev_end)
            q.put(curr_end)
        return True if q.qsize() == 1 else False
    
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key=lambda x:x[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            end = res[-1][1]
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if curr_start <= end:
                res[-1][1] = max(curr_end, end)
            else:
                res.append(intervals[i])
                end = curr_end
        return res
    
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        res = []
        for i in range(len(intervals)):
            if intervals[i][1] < newInterval[0]:
                res.append(intervals[i])
            elif intervals[i][0] > newInterval[1]:
                res.append(newInterval)
                newInterval = intervals[i]
            elif intervals[i][1] >= newInterval[0]:
                newInterval[0] = min(intervals[i][0], newInterval[0])
                newInterval[1] = max(intervals[i][1], newInterval[1])  
        res.append(newInterval)
        return res
    
    def intervalIntersection(self, firstList, secondList):
        """
        :type firstList: List[List[int]]
        :type secondList: List[List[int]]
        :rtype: List[List[int]]
        """
        i = 0
        j = 0
        res = []
        while i < len(firstList) and j < len(secondList):
            first_start = firstList[i][0]
            first_end = firstList[i][1]
            second_start = secondList[j][0]
            second_end = secondList[j][1]
            if first_start <= second_end and second_start <= first_end:
                res.append([max(first_start, second_start), min(first_end, second_end)])
            if first_end <= second_end:
                i += 1
            else:
                j += 1
        return res
    
    def employeeFreeTime(self, schedule):
        """
        :type schedule: [[Interval]]
        :rtype: [Interval]
        """
        intervals = []
        for employee in schedule:
            for interval in employee:
                intervals.append(interval)
        intervals.sort(key = lambda interval: interval.start)

        freeTime = []
        end = intervals[0].end
        for i in range(1, len(intervals)):
            start = intervals[i].start
            if start > end:
                freeTime.append(Interval(start = end, end = start))
                end = intervals[i].end
            else:
                end = max(intervals[i].end, end)
        return freeTime

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head
        while curr:
            n = curr.next
            curr.next = prev
            prev = curr
            curr = n
        return prev
    
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        seen = set()
        while headA:
            seen.add(headA)
            headA = headA.next
        while headB:
            if headB in seen:
                return headB
            headB = headB.next
        return None
    
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        first = head
        last = head
        for i in range(1, k):
            first = first.next
            
        null_checker = first 
        while null_checker.next:
            last = last.next
            null_checker = null_checker.next
        first.val, last.val = last.val, first.val
        return head

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        jump = dummy
        dummy.next = head
        left = right = head

        while True:
            count = 0
            while right and count < k:
                count += 1
                right = right.next
            
            if count == k:
                prev = right
                curr = left
                for i in range(k):
                    nxt = curr.next
                    curr.next = prev
                    prev = curr
                    curr = nxt
                jump.next = prev
                jump = left
                left = right

            else:
                return dummy.next

    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        prev = dummy
        dummy.next = head
        for i in range(left-1):
            prev = prev.next
        curr = prev.next
        tail = None

        for i in range(right-left+1):
            nxt = curr.next
            curr.next = tail
            tail = curr
            curr = nxt
        
        prev.next.next = curr
        prev.next = tail

        return dummy.next
    
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        while head and head.next:
            tmp = head.next
            head.next = tmp.next
            tmp.next = head
            p.next = tmp
            
            head = head.next
            p = tmp.next
        return dummy.next
    
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        heap = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i))
        
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, i = heapq.heappop(heap)
            curr.next = lists[i]
            curr = curr.next
            if lists[i].next:
                lists[i] = lists[i].next
                heapq.heappush(heap, (lists[i].val, i))
        return dummy.next
